"""
The main VIT-EAR model: an ordinal, cost-sensitive, multi-output classifier trained
on the synthetic dataset from synthetic_data.py. Results here are on fabricated data
and show the pipeline works end to end - they aren't evidence about real students.

This script:
  1. Trains a plain multiclass model with no cost-awareness as an in-house baseline
     (separate from the literature baseline in baseline_model.py).
  2. Trains the cost-sensitive version: instance re-weighting during training and a
     Bayes-risk decision rule at inference time, using the 4x4 cost matrix below.
  3. Compares the two on accuracy, macro-F1, per-level recall, and expected cost.
  4. Computes SHAP values for the cost-sensitive model and aggregates them into
     source clusters to get a recommended intervention type per at-risk student.
  5. Checks that routing against the known synthetic ground truth (only possible
     here because we generated the data and know the true driver for each student).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

RISK_LEVELS = {0: "No Risk", 1: "Backlog/Probation Risk", 2: "Attendance-Debarment Risk", 3: "Withdrawal Risk"}

# cost grows with |true - pred|, and missing severity (under-prediction) costs
# more than a false alarm (over-prediction)
W_UNDER, W_OVER, P = 3.0, 1.0, 1.0


def build_cost_matrix() -> np.ndarray:
    C = np.zeros((4, 4))
    for true in range(4):
        for pred in range(4):
            under = max(0, true - pred) ** P
            over = max(0, pred - true) ** P
            C[true, pred] = W_UNDER * under + W_OVER * over
    return C


# maps each feature to one of the 7 source clusters used in synthetic_data.py
# (there's an 8th cluster, "counselling", but it has no features - it's just the
# mentor-log label itself)
FEATURE_CLUSTERS = {
    "cgpa_cumulative": "academic", "backlogs_count": "academic", "backlogs_age_max_weeks": "academic",
    "entrance_percentile": "academic", "branch": "academic", "admission_category": "academic",
    "home_state": "academic", "schooling_medium": "academic", "first_gen_college_goer": "academic",
    "mean_attendance_pct": "attendance", "min_attendance_pct": "attendance",
    "consecutive_absence_streak_max": "attendance",
    "attendance_trend": "attendance", "timing_of_decline": "attendance",
    "max_possible_attendance_pct": "attendance",
    "avg_pace_score": "pedagogy", "avg_clarity_score": "pedagogy",
    "avg_doubt_resolution_score": "pedagogy", "avg_teaching_style_fit_score": "pedagogy",
    "diagnostic_content_score": "diagnostic", "diagnostic_delivery_sensitive_score": "diagnostic",
    "avg_session_engagement_exposure": "visual_engagement",
    "vtop_login_freq_per_week": "behavioural", "submission_timing_relative": "behavioural",
    "footfall_library_freq_per_week": "behavioural", "hostel_curfew_violations": "behavioural",
    "hosteller_flag": "behavioural",
    "fee_payment_status": "financial", "scholarship_category": "financial",
    "education_loan_status": "financial", "family_income_bracket": "financial",
}

CLUSTER_TO_INTERVENTION = {
    "academic": "academic", "attendance": "academic", "pedagogy": "pedagogical",
    "diagnostic": "pedagogical", "visual_engagement": "pedagogical",
    "behavioural": "psychosocial", "financial": "financial",
}

# fields left out of the model on purpose:
#   student_id - just an identifier
#   social_category - used for fairness checks only, never as a predictive input
#   latent_risk_score - this is literally what generated the label, so it's leakage
#   true_dominant_cluster - ground truth, only used to validate the SHAP routing
#   risk_level_label - just a string version of the target
#   mentor_flag_category - the z_i label itself, used for validation not as a feature
#   attendance_debarment_threshold / condonation bands - constant across every row
#   classes_total_in_sem, classes_conducted_so_far, classes_remaining_in_sem - these only
#     exist to build max_possible_attendance_pct in synthetic_data.py. feeding all three
#     plus the derived ratio would be redundant and would inflate the attendance
#     cluster's SHAP total for no real reason, so only the derived ratio is kept
EXCLUDED = {
    "student_id", "social_category", "latent_risk_score", "true_dominant_cluster",
    "risk_level_label", "mentor_flag_category", "risk_level",
    "attendance_debarment_threshold", "attendance_condonation_band_low", "attendance_condonation_band_high",
    "classes_total_in_sem", "classes_conducted_so_far", "classes_remaining_in_sem",
}


def load_data():
    df = pd.read_csv("data/synthetic/vit_ear_synthetic_v1.csv")
    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    X = df[feature_cols].copy()
    y = df["risk_level"].copy()
    # pandas >= 3.0 defaults CSV string columns to its new "str" extension dtype rather
    # than legacy numpy `object`, so check both via is_object_dtype/is_string_dtype.
    from pandas.api.types import is_object_dtype, is_string_dtype
    cat_cols = [c for c in X.columns if is_object_dtype(X[c]) or is_string_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "NotApplicable").astype(str)
    return df, X, y, cat_cols


def bayes_risk_predict(proba: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """Picks the class that minimizes expected cost given the model's probabilities,
    instead of just taking the highest-probability class. Generalizes threshold
    shifting to more than two classes."""
    expected_cost = proba @ cost_matrix  # (n_samples, n_pred_classes)
    return expected_cost.argmin(axis=1)


def total_expected_cost(y_true, y_pred, cost_matrix) -> float:
    return float(cost_matrix[y_true, y_pred].mean())


def raw_cluster_totals(sample_shap_for_pred_class: np.ndarray, feature_names: list, cluster_names: list) -> np.ndarray:
    """Sums |SHAP| per cluster for one sample, returned as a vector aligned to cluster_names."""
    totals = np.zeros(len(cluster_names))
    for feat_i, feat_name in enumerate(feature_names):
        cluster = FEATURE_CLUSTERS.get(feat_name)
        if cluster is not None:
            totals[cluster_names.index(cluster)] += abs(sample_shap_for_pred_class[feat_i])
    return totals


def cv_routing_validation(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, cat_cols: list,
                           cost_matrix: np.ndarray, n_folds: int = 5, random_state: int = 42) -> dict:
    """Runs 5-fold CV and pools the at-risk predictions across all folds before scoring.
    A single split only has about 30 at-risk students, which isn't enough to measure a
    7-way classification accuracy reliably (this number swung between 13% and 50%
    across different feature choices during testing, purely from sample noise). Pooling
    across folds gives about 150 at-risk students total, which is much more stable."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cluster_names = sorted(set(FEATURE_CLUSTERS.values()))
    all_cluster_match, all_intervention_match = [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        sw = 1.0 + cost_matrix[y_tr, 0]
        model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, loss_function="MultiClass",
                                    random_seed=random_state, verbose=False, cat_features=cat_cols, thread_count=4)
        model.fit(X_tr, y_tr, sample_weight=sw)
        proba = model.predict_proba(X_te)
        pred = bayes_risk_predict(proba, cost_matrix)
        pred = apply_deterministic_overrides(X_te, pred)

        explainer = shap.TreeExplainer(model)
        shap_arr = np.asarray(explainer.shap_values(X_te))
        if shap_arr.shape[0] == 4 and shap_arr.shape[1] == len(X_te):
            shap_arr = np.transpose(shap_arr, (1, 2, 0))

        fold_df = df.iloc[te_idx].reset_index(drop=True)
        feature_names = list(X_te.columns)
        for row_i, pred_class in enumerate(pred):
            if fold_df.loc[row_i, "risk_level"] == 0:
                continue  # no intervention needed, nothing to score here
            if pred_class == 0:
                pred_cluster, pred_intervention = "none", "none"
            else:
                totals = raw_cluster_totals(shap_arr[row_i, :, pred_class], feature_names, cluster_names)
                pred_cluster = cluster_names[int(np.argmax(totals))]
                pred_intervention = CLUSTER_TO_INTERVENTION[pred_cluster]
            all_cluster_match.append(pred_cluster == fold_df.loc[row_i, "true_dominant_cluster"])
            all_intervention_match.append(pred_intervention == fold_df.loc[row_i, "mentor_flag_category"])

    return {
        "n_at_risk_pooled": len(all_cluster_match),
        "dominant_cluster_recovery_accuracy": float(np.mean(all_cluster_match)),
        "intervention_type_match_accuracy": float(np.mean(all_intervention_match)),
    }


def apply_deterministic_overrides(X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Some outcomes are certain rather than statistical, so we don't leave them for the
    model to infer. If a student can't reach the attendance-debarment threshold even with
    perfect attendance from here on, this floors their predicted risk at Level 2 no matter
    what the model itself says. Applied the same way to both the plain and cost-sensitive
    models."""
    pred = pred.copy()
    if "attendance_debarment_mathematically_certain" in X.columns:
        certain = X["attendance_debarment_mathematically_certain"].to_numpy() == 1
        pred[certain & (pred < 2)] = 2
    return pred


def evaluate(name, y_true, y_pred, cost_matrix):
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    for lvl in range(4):
        mask = y_true == lvl
        rec = recall_score(y_true, y_pred, labels=[lvl], average="macro") if mask.any() else float("nan")
        print(f"  Recall on Level {lvl} ({RISK_LEVELS[lvl]}): {rec:.4f}  (n={mask.sum()})")
    cost = total_expected_cost(y_true, y_pred, cost_matrix)
    print(f"Mean expected cost per student: {cost:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mean_cost": cost,
    }


def main():
    df, X, y, cat_cols = load_data()
    cost_matrix = build_cost_matrix()
    print("Cost matrix C[true, pred]:\n", cost_matrix)

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train].values, y.iloc[idx_test].values

    # plain in-house baseline, no cost-awareness
    plain_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05, loss_function="MultiClass",
        random_seed=42, verbose=False, cat_features=cat_cols, thread_count=4,
    )
    plain_model.fit(X_train, y_train)
    plain_proba = plain_model.predict_proba(X_test)
    plain_pred = plain_proba.argmax(axis=1)  # plain arg-max, equivalent to a 0/1 cost matrix
    plain_pred = apply_deterministic_overrides(X_test, plain_pred)
    plain_metrics = evaluate("plain model (arg-max decision)", y_test, plain_pred, cost_matrix)

    # cost-sensitive model: weight each example by the cost of missing that severity
    # level entirely (i.e. predicting "No Risk" when the truth is worse)
    sample_weight = 1.0 + cost_matrix[y_train, 0]
    cs_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05, loss_function="MultiClass",
        random_seed=42, verbose=False, cat_features=cat_cols, thread_count=4,
    )
    cs_model.fit(X_train, y_train, sample_weight=sample_weight)
    cs_proba = cs_model.predict_proba(X_test)
    cs_pred = bayes_risk_predict(cs_proba, cost_matrix)  # calibrated decision rule, not arg-max
    cs_pred = apply_deterministic_overrides(X_test, cs_pred)
    cs_metrics = evaluate("cost-sensitive model (Bayes-risk decision)", y_test, cs_pred, cost_matrix)

    print("\n=== comparison ===")
    print(f"Mean expected cost - plain: {plain_metrics['mean_cost']:.4f}  "
          f"vs cost-sensitive: {cs_metrics['mean_cost']:.4f}  "
          f"({'improved' if cs_metrics['mean_cost'] < plain_metrics['mean_cost'] else 'no improvement'})")

    # SHAP source-cluster routing, computed on the cost-sensitive model
    print("\nComputing SHAP values for source-cluster routing...")
    explainer = shap.TreeExplainer(cs_model)
    shap_out = explainer.shap_values(X_test)
    shap_arr = np.asarray(shap_out)
    print("Raw SHAP output shape:", shap_arr.shape)
    # shap's output layout has changed across versions, so handle both cases and
    # normalize to (n_samples, n_features, n_classes)
    if shap_arr.shape[0] == 4 and shap_arr.shape[1] == len(X_test):
        shap_arr = np.transpose(shap_arr, (1, 2, 0))
    elif shap_arr.ndim == 3 and shap_arr.shape[-1] == 4:
        pass
    else:
        raise RuntimeError(f"Unexpected SHAP output shape {shap_arr.shape}; inspect and adjust.")

    feature_names = list(X_test.columns)
    cluster_names = sorted(set(FEATURE_CLUSTERS.values()))

    # routing uses a plain sum of |SHAP| per cluster, no normalization. tried a few
    # normalization schemes (per-feature mean, min-max on the full test set, min-max on
    # just the at-risk subset) to account for clusters having different feature counts,
    # but all of them were less stable than plain sum at this sample size - they just
    # traded one cluster's over-representation for another's. the actual fix was
    # upstream: pruning the redundant attendance features (see FEATURE_CLUSTERS) so no
    # cluster is inflated by feature count to begin with, which makes plain sum fair
    routed_intervention, routed_cluster = [], []
    for row_i, pred_class in enumerate(cs_pred):
        if pred_class == 0:
            routed_cluster.append("none")
            routed_intervention.append("none")
        else:
            totals = raw_cluster_totals(shap_arr[row_i, :, pred_class], feature_names, cluster_names)
            top_cluster = cluster_names[int(np.argmax(totals))]
            routed_cluster.append(top_cluster)
            routed_intervention.append(CLUSTER_TO_INTERVENTION[top_cluster])

    test_df = df.iloc[idx_test].reset_index(drop=True)
    test_df["pred_risk_level"] = cs_pred
    test_df["pred_intervention_type"] = routed_intervention
    test_df["pred_dominant_cluster"] = routed_cluster

    # check routing against the known synthetic ground truth
    at_risk_mask = test_df["risk_level"] > 0
    cluster_match = (test_df.loc[at_risk_mask, "pred_dominant_cluster"]
                      == test_df.loc[at_risk_mask, "true_dominant_cluster"])
    intervention_match = (test_df.loc[at_risk_mask, "pred_intervention_type"]
                           == test_df.loc[at_risk_mask, "mentor_flag_category"])
    print(f"\n=== routing validation, single split (n={at_risk_mask.sum()}) ===")
    print(f"Dominant-cluster recovery accuracy: {cluster_match.mean():.4f}")
    print(f"Intervention-type match accuracy:   {intervention_match.mean():.4f}")
    print("(this is noisy at such a small sample size, see the pooled 5-fold number below)")

    print("\nRunning 5-fold CV routing validation (pooled)...")
    cv_routing = cv_routing_validation(df, X, y, cat_cols, cost_matrix)
    print(f"=== routing validation, pooled 5-fold CV "
          f"(n={cv_routing['n_at_risk_pooled']}) ===")
    print(f"Dominant-cluster recovery accuracy: {cv_routing['dominant_cluster_recovery_accuracy']:.4f}")
    print(f"Intervention-type match accuracy:   {cv_routing['intervention_type_match_accuracy']:.4f}")

    Path("results").mkdir(exist_ok=True)
    results = {
        "provenance": "Synthetic data only. Shows the mechanism works, not a claim about real students.",
        "cost_matrix": cost_matrix.tolist(),
        "plain_model": plain_metrics,
        "cost_sensitive_model": cs_metrics,
        "cost_improvement": plain_metrics["mean_cost"] - cs_metrics["mean_cost"],
        "shap_routing_validation_single_split": {
            "n_at_risk_test_samples": int(at_risk_mask.sum()),
            "dominant_cluster_recovery_accuracy": float(cluster_match.mean()),
            "intervention_type_match_accuracy": float(intervention_match.mean()),
            "note": "Noisy at this sample size, see shap_routing_validation_cv_pooled instead.",
        },
        "shap_routing_validation_cv_pooled": cv_routing,
    }
    with open("results/cost_sensitive_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    test_df.to_csv("results/test_predictions_with_routing.csv", index=False)
    print("\nSaved results/cost_sensitive_model_results.json")
    print("Saved results/test_predictions_with_routing.csv")


if __name__ == "__main__":
    main()
