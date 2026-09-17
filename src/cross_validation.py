"""
k-fold cross-validation and Optuna hyperparameter search, run on both the
literature-benchmark baseline (real UCI data) and the VIT-EAR cost-sensitive
prototype (synthetic data).

This is a dry run of the validation approach on data available now, so the
cross-validation and hyperparameter search machinery is already working correctly
once real data is available.
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from cost_sensitive_model import (
    build_cost_matrix, bayes_risk_predict, total_expected_cost, load_data as load_synthetic_data,
    apply_deterministic_overrides,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_FOLDS = 5
RANDOM_STATE = 42


# --- Part 1: k-fold CV + hyperparameter search on the real UCI baseline ---
def load_uci():
    df = pd.read_csv("data/raw/uci_dropout_academic_success.csv")
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    y = (df["Target"] == "Dropout").astype(int).values
    X = df.drop(columns=["Target"])
    return X, y


def cv_baseline(X, y, params=None, n_folds=N_FOLDS):
    params = params or dict(iterations=300, depth=6, learning_rate=0.05)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr_res, y_tr_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
        model = CatBoostClassifier(**params, loss_function="Logloss", random_seed=RANDOM_STATE, verbose=False, thread_count=4)
        model.fit(X_tr_res, y_tr_res)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(dict(
            fold=fold, accuracy=accuracy_score(y_te, pred),
            f1=f1_score(y_te, pred), auc=roc_auc_score(y_te, proba),
        ))
    dfm = pd.DataFrame(fold_metrics)
    summary = {f"{c}_mean": dfm[c].mean() for c in ["accuracy", "f1", "auc"]}
    summary.update({f"{c}_std": dfm[c].std() for c in ["accuracy", "f1", "auc"]})
    return dfm, summary


def optuna_search_baseline(X, y, n_trials=25):
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 500, step=50),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        )
        _, summary = cv_baseline(X, y, params=params, n_folds=3)  # 3-fold inside search to save time
        return summary["f1_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# --- Part 2: k-fold CV + hyperparameter search on the synthetic-data prototype ---
def cv_cost_sensitive(X, y, cat_cols, cost_matrix, params=None, n_folds=N_FOLDS):
    params = params or dict(iterations=300, depth=6, learning_rate=0.05)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        sample_weight = 1.0 + cost_matrix[y_tr, 0]
        model = CatBoostClassifier(
            **params, loss_function="MultiClass", random_seed=RANDOM_STATE, thread_count=4,
            verbose=False, cat_features=cat_cols,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        proba = model.predict_proba(X_te)
        pred = bayes_risk_predict(proba, cost_matrix)
        pred = apply_deterministic_overrides(X_te, pred)
        fold_metrics.append(dict(
            fold=fold,
            accuracy=accuracy_score(y_te, pred),
            macro_f1=f1_score(y_te, pred, average="macro"),
            mean_cost=total_expected_cost(y_te, pred, cost_matrix),
        ))
    dfm = pd.DataFrame(fold_metrics)
    summary = {f"{c}_mean": dfm[c].mean() for c in ["accuracy", "macro_f1", "mean_cost"]}
    summary.update({f"{c}_std": dfm[c].std() for c in ["accuracy", "macro_f1", "mean_cost"]})
    return dfm, summary


def optuna_search_cost_sensitive(X, y, cat_cols, cost_matrix, n_trials=25):
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 500, step=50),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        )
        _, summary = cv_cost_sensitive(X, y, cat_cols, cost_matrix, params=params, n_folds=3)
        return summary["mean_cost_mean"]  # minimizing expected cost here, not maximizing accuracy

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def main():
    results = {}

    print("=" * 70)
    print("Part 1: literature-benchmark baseline (real UCI data)")
    print("=" * 70)
    X_uci, y_uci = load_uci()
    dfm, summary = cv_baseline(X_uci, y_uci)
    print(f"\n{N_FOLDS}-fold CV (default params):")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    results["baseline_cv_default"] = summary

    print(f"\nRunning Optuna hyperparameter search (25 trials, 3-fold inner CV)...")
    study = optuna_search_baseline(X_uci, y_uci, n_trials=25)
    print(f"Best F1 (3-fold): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Re-validating best params with full {N_FOLDS}-fold CV...")
    _, best_summary = cv_baseline(X_uci, y_uci, params=study.best_params)
    for k, v in best_summary.items():
        print(f"  {k}: {v:.4f}")
    results["baseline_optuna_best_params"] = study.best_params
    results["baseline_cv_tuned"] = best_summary

    print("\n" + "=" * 70)
    print("Part 2: VIT-EAR cost-sensitive prototype (synthetic data)")
    print("=" * 70)
    _, X_syn, y_syn, cat_cols = load_synthetic_data()
    cost_matrix = build_cost_matrix()

    dfm2, summary2 = cv_cost_sensitive(X_syn, y_syn, cat_cols, cost_matrix)
    print(f"\n{N_FOLDS}-fold CV (default params):")
    for k, v in summary2.items():
        print(f"  {k}: {v:.4f}")
    results["cost_sensitive_cv_default"] = summary2

    print(f"\nRunning Optuna hyperparameter search (25 trials, 3-fold inner CV, minimizing mean cost)...")
    study2 = optuna_search_cost_sensitive(X_syn, y_syn, cat_cols, cost_matrix, n_trials=25)
    print(f"Best mean cost (3-fold): {study2.best_value:.4f}")
    print(f"Best params: {study2.best_params}")
    print(f"Re-validating best params with full {N_FOLDS}-fold CV...")
    _, best_summary2 = cv_cost_sensitive(X_syn, y_syn, cat_cols, cost_matrix, params=study2.best_params)
    for k, v in best_summary2.items():
        print(f"  {k}: {v:.4f}")
    results["cost_sensitive_optuna_best_params"] = study2.best_params
    results["cost_sensitive_cv_tuned"] = best_summary2

    Path("results").mkdir(exist_ok=True)
    with open("results/cross_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved results/cross_validation_results.json")


if __name__ == "__main__":
    main()
