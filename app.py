"""
VIT-EAR mentor dashboard - an interactive prototype of the mentor-facing
explanation tool described in the project writeup.

Runs entirely on synthetic data (data/synthetic/vit_ear_synthetic_v1.csv). Every
number and prediction shown here demonstrates that the modeling mechanism works
end to end, not a claim about real students.

Run with:  streamlit run app.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from catboost import CatBoostClassifier
from PIL import Image

# Every relative path below (data/, results/) assumes cwd == this file's directory,
# which is not guaranteed depending on how `streamlit run` was invoked.
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cost_sensitive_model import (  # noqa: E402
    build_cost_matrix, bayes_risk_predict, load_data, FEATURE_CLUSTERS,
    CLUSTER_TO_INTERVENTION, RISK_LEVELS, apply_deterministic_overrides, raw_cluster_totals,
)
from engagement_pipeline import (  # noqa: E402
    EXTRACT_DIR, MACRO_CATEGORY, featurize, normalize_label, ensure_extracted,
)

st.set_page_config(page_title="VIT-EAR Mentor Dashboard (Prototype)", layout="wide")

RISK_COLORS = {0: "#2E7D32", 1: "#F9A825", 2: "#EF6C00", 3: "#C62828"}


# ---------------------------------------------------------------------------
# Cached data + model loading (trained once per Streamlit session, not per click)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    df, X, y, cat_cols = load_data()
    return df, X, y, cat_cols


@st.cache_resource
def get_models(_X: pd.DataFrame, _y: pd.Series, cat_cols: list):
    # trains a smaller model than the standalone script (150 iterations vs 300) so
    # the dashboard loads quickly. the reported numbers in results/ come from the
    # full 300-iteration model in cost_sensitive_model.py, not this one.
    cost_matrix = build_cost_matrix()
    plain = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.08,
                                loss_function="MultiClass", random_seed=42, thread_count=2,
                                verbose=False, cat_features=cat_cols)
    plain.fit(_X, _y)

    sample_weight = 1.0 + cost_matrix[_y.values, 0]
    cs = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.08,
                             loss_function="MultiClass", random_seed=42, thread_count=2,
                             verbose=False, cat_features=cat_cols)
    cs.fit(_X, _y, sample_weight=sample_weight)

    explainer = shap.TreeExplainer(cs)
    return plain, cs, explainer, cost_matrix


def normalize_shap_shape(shap_out, n_rows):
    arr = np.asarray(shap_out)
    if arr.shape[0] == 4 and arr.shape[1] == n_rows:
        arr = np.transpose(arr, (1, 2, 0))
    return arr  # (n_rows, n_features, n_classes)


def cluster_contributions(shap_row_for_class: np.ndarray, feature_names: list) -> dict:
    # plain sum of |SHAP| per cluster - see cost_sensitive_model.py for why this
    # ended up being the version used instead of a normalized one
    cluster_names = sorted(set(FEATURE_CLUSTERS.values()))
    totals = raw_cluster_totals(shap_row_for_class, feature_names, cluster_names)
    return dict(zip(cluster_names, totals))


def route_intervention(pred_class: int, cluster_totals: dict):
    if pred_class == 0:
        return "none", None
    top_cluster = max(cluster_totals, key=cluster_totals.get)
    return CLUSTER_TO_INTERVENTION[top_cluster], top_cluster


def render_prediction_panel(label, pred_class, cluster_totals, intervention, top_cluster):
    color = RISK_COLORS[pred_class]
    st.markdown(f"#### {label}")
    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background:{color}22;"
        f"border-left:6px solid {color};'>"
        f"<b>Predicted risk level:</b> {pred_class} ({RISK_LEVELS[pred_class]})<br>"
        f"<b>Recommended intervention:</b> {intervention}"
        + (f" (driven by: {top_cluster})" if top_cluster else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if cluster_totals:
        fig = go.Figure(go.Bar(
            x=list(cluster_totals.values()), y=list(cluster_totals.keys()),
            orientation="h", marker_color=color,
        ))
        fig.update_layout(
            title="SHAP contribution by source cluster (toward the predicted class)",
            xaxis_title="Sum of |SHAP value|", height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.title("VIT-EAR Mentor Dashboard")
st.caption(
    "All data and predictions on this page are synthetic "
    "(see data/synthetic/vit_ear_synthetic_v1.meta.json). This shows the "
    "modeling pipeline works end to end, not a claim about real students."
)

df, X, y, cat_cols = get_data()
with st.spinner("Training models (first load only, cached after this, roughly 30-60s here)..."):
    plain_model, cs_model, explainer, cost_matrix = get_models(X, y, cat_cols)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Student Explorer", "🧪 What-If Simulator", "📊 Model Performance", "🎥 Engagement Pipeline"]
)

# ============================== Tab 1: Student Explorer ==============================
with tab1:
    st.subheader("Pick a synthetic student")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        filter_level = st.selectbox(
            "Filter by true synthetic risk level", ["All"] + [f"{k} - {v}" for k, v in RISK_LEVELS.items()]
        )
        pool = df if filter_level == "All" else df[df["risk_level"] == int(filter_level.split(" - ")[0])]
        student_id = st.selectbox("Student ID", pool["student_id"].tolist())

    row = df[df["student_id"] == student_id].iloc[0]
    row_X = X[df["student_id"] == student_id]

    with col_b:
        st.write("**Key profile fields**")
        st.table(pd.DataFrame({
            "Field": ["CGPA", "Backlogs", "Mean attendance %", "Max possible attendance % (remaining sem)",
                      "Debarment mathematically certain?", "Fee status", "Scholarship",
                      "True (synthetic) risk level", "True dominant driver (ground truth)"],
            "Value": [row["cgpa_cumulative"], row["backlogs_count"], row["mean_attendance_pct"],
                      row["max_possible_attendance_pct"],
                      "Yes" if row["attendance_debarment_mathematically_certain"] == 1 else "No",
                      row["fee_payment_status"], row["scholarship_category"],
                      f"{row['risk_level']} - {row['risk_level_label']}", row["true_dominant_cluster"]],
        }).set_index("Field"))

    st.divider()
    left, right = st.columns(2)

    # Plain model
    plain_proba = plain_model.predict_proba(row_X)[0]
    plain_pred_raw = int(np.argmax(plain_proba))
    plain_pred = int(apply_deterministic_overrides(row_X, np.array([plain_pred_raw]))[0])
    plain_shap = normalize_shap_shape(explainer.shap_values(row_X), 1)[0, :, plain_pred]
    plain_clusters = cluster_contributions(plain_shap, list(X.columns))
    plain_intervention, plain_top = route_intervention(plain_pred, plain_clusters if plain_pred else None)
    if plain_pred != plain_pred_raw:
        plain_intervention, plain_top = "academic", "attendance"
    with left:
        if plain_pred != plain_pred_raw:
            st.warning(f"Deterministic override: model predicted Level {plain_pred_raw}, but debarment "
                       f"is mathematically certain given remaining classes, so it's floored to Level 2.")
        render_prediction_panel("Plain model (arg-max)", plain_pred,
                                 plain_clusters if plain_pred else {}, plain_intervention, plain_top)

    # Cost-sensitive model
    cs_proba = cs_model.predict_proba(row_X)[0]
    cs_pred_raw = int(bayes_risk_predict(cs_proba.reshape(1, -1), cost_matrix)[0])
    cs_pred = int(apply_deterministic_overrides(row_X, np.array([cs_pred_raw]))[0])
    cs_shap = normalize_shap_shape(explainer.shap_values(row_X), 1)[0, :, cs_pred]
    cs_clusters = cluster_contributions(cs_shap, list(X.columns))
    cs_intervention, cs_top = route_intervention(cs_pred, cs_clusters if cs_pred else None)
    if cs_pred != cs_pred_raw:
        cs_intervention, cs_top = "academic", "attendance"
    with right:
        if cs_pred != cs_pred_raw:
            st.warning(f"Deterministic override: model predicted Level {cs_pred_raw}, but debarment "
                       f"is mathematically certain given remaining classes, so it's floored to Level 2.")
        render_prediction_panel("Cost-sensitive model (Bayes-risk decision)", cs_pred,
                                 cs_clusters if cs_pred else {}, cs_intervention, cs_top)

    if plain_pred != cs_pred:
        st.info(
            f"The two models disagree on this student: plain predicts Level {plain_pred} "
            f"({RISK_LEVELS[plain_pred]}), cost-sensitive predicts Level {cs_pred} "
            f"({RISK_LEVELS[cs_pred]}). This is the intended effect of the cost matrix: "
            f"the cost-sensitive model is willing to flag more students to avoid missing severe cases."
        )

# ============================== Tab 2: What-If Simulator ==============================
with tab2:
    st.subheader("Build a hypothetical student and see the prediction update live")
    template = X.iloc[[0]].copy()  # start from a real synthetic row, then override key fields

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Academic**")
        cgpa = st.slider("CGPA (10-point scale)", 0.0, 10.0, float(template["cgpa_cumulative"].iloc[0]), 0.1)
        backlogs = st.slider("Backlogs count", 0, 8, int(template["backlogs_count"].iloc[0]))
    with c2:
        st.markdown("**Attendance & Pedagogy**")
        attendance = st.slider("Mean attendance % (so far)", 30.0, 100.0, float(template["mean_attendance_pct"].iloc[0]))
        classes_conducted = st.slider("Classes conducted so far (out of 50 in the semester)", 5, 49, 25)
        teaching_fit = st.slider("Teaching-style fit score (1-5)", 1.0, 5.0,
                                  float(template["avg_teaching_style_fit_score"].iloc[0]))
        diagnostic = st.slider("Diagnostic content score (0-1)", 0.0, 1.0,
                                float(template["diagnostic_content_score"].iloc[0]))
    with c3:
        st.markdown("**Financial & Behavioural**")
        fee_status = st.selectbox("Fee payment status",
                                   ["Fully Paid", "On Installment (On-Time)",
                                    "On Installment (Overdue)", "Fee Due >30 Days"])
        scholarship = st.selectbox("Scholarship category",
                                    ["No Scholarship", "Merit-Based", "Category-Based",
                                     "Sports-Cultural Quota", "Institutional Need-Based"])
        engagement = st.slider("Session engagement exposure (0-1)", 0.0, 1.0,
                                float(template["avg_session_engagement_exposure"].iloc[0]))

    classes_remaining = 50 - classes_conducted
    classes_attended = round(attendance / 100 * classes_conducted)
    max_possible_pct = (classes_attended + classes_remaining) / 50 * 100
    debarment_certain = 1 if max_possible_pct < 65.0 else 0

    custom = template.copy()
    custom["cgpa_cumulative"] = cgpa
    custom["backlogs_count"] = backlogs
    custom["mean_attendance_pct"] = attendance
    custom["max_possible_attendance_pct"] = max_possible_pct
    custom["attendance_debarment_mathematically_certain"] = debarment_certain
    custom["avg_teaching_style_fit_score"] = teaching_fit
    custom["diagnostic_content_score"] = diagnostic
    custom["fee_payment_status"] = fee_status
    custom["scholarship_category"] = scholarship
    custom["avg_session_engagement_exposure"] = engagement

    st.divider()
    st.metric("Max possible attendance % (perfect attendance from here on)", f"{max_possible_pct:.1f}%")
    if debarment_certain:
        st.error(
            "Even attending every remaining class, this student cannot reach the 65% "
            "debarment threshold. This is a mathematical certainty, not a prediction, "
            "so both models below are floored to at least Level 2 regardless of their "
            "own output."
        )
    wcol1, wcol2 = st.columns(2)

    w_plain_proba = plain_model.predict_proba(custom)[0]
    w_plain_pred_raw = int(np.argmax(w_plain_proba))
    w_plain_pred = int(apply_deterministic_overrides(custom, np.array([w_plain_pred_raw]))[0])
    w_plain_shap = normalize_shap_shape(explainer.shap_values(custom), 1)[0, :, w_plain_pred]
    w_plain_clusters = cluster_contributions(w_plain_shap, list(X.columns))
    w_plain_intervention, w_plain_top = route_intervention(w_plain_pred, w_plain_clusters if w_plain_pred else None)
    if w_plain_pred != w_plain_pred_raw:
        w_plain_intervention, w_plain_top = "academic", "attendance"
    with wcol1:
        render_prediction_panel("Plain model", w_plain_pred,
                                 w_plain_clusters if w_plain_pred else {}, w_plain_intervention, w_plain_top)

    w_cs_proba = cs_model.predict_proba(custom)[0]
    w_cs_pred_raw = int(bayes_risk_predict(w_cs_proba.reshape(1, -1), cost_matrix)[0])
    w_cs_pred = int(apply_deterministic_overrides(custom, np.array([w_cs_pred_raw]))[0])
    w_cs_shap = normalize_shap_shape(explainer.shap_values(custom), 1)[0, :, w_cs_pred]
    w_cs_clusters = cluster_contributions(w_cs_shap, list(X.columns))
    w_cs_intervention, w_cs_top = route_intervention(w_cs_pred, w_cs_clusters if w_cs_pred else None)
    if w_cs_pred != w_cs_pred_raw:
        w_cs_intervention, w_cs_top = "academic", "attendance"
    with wcol2:
        render_prediction_panel("Cost-sensitive model", w_cs_pred,
                                 w_cs_clusters if w_cs_pred else {}, w_cs_intervention, w_cs_top)

# ============================== Tab 3: Model Performance ==============================
with tab3:
    st.subheader("Headline results so far")
    st.caption("Pulled from results/*.json. Run the src/ scripts to regenerate after any change.")

    results_dir = Path("results")
    cols = st.columns(3)

    def show_json(col, title, filename, keys=None):
        path = results_dir / filename
        with col:
            st.markdown(f"**{title}**")
            if path.exists():
                data = json.loads(path.read_text())
                if keys:
                    data = {k: data[k] for k in keys if k in data}
                st.json(data)
            else:
                st.warning(f"{filename} not found. Run the corresponding script first.")

    show_json(cols[0], "Literature-benchmark baseline", "baseline_literature_benchmark.json")
    show_json(cols[1], "Cost-sensitive model (single split)", "cost_sensitive_model_results.json",
               keys=["plain_model", "cost_sensitive_model", "cost_improvement", "shap_routing_validation"])
    show_json(cols[2], "Cross-validation + tuning", "cross_validation_results.json",
               keys=["cost_sensitive_cv_default", "cost_sensitive_cv_tuned"])
    show_json(cols[0], "Engagement pipeline", "engagement_pipeline_results.json")

# ============================== Tab 4: Engagement Pipeline ==============================
with tab4:
    st.subheader("Camera-engagement signal")
    st.caption(
        "Real data (Kaggle 'Student-Engagement-Dataset', CC0), used as a substitute for "
        "DAiSEE, which needs a manual license-agreement request that couldn't be arranged "
        "in time. The images below are from a public research dataset, not real students."
    )

    eng_results_path = Path("results/engagement_pipeline_results.json")
    if not eng_results_path.exists() or not EXTRACT_DIR.exists():
        st.warning(
            "Run `python src/engagement_pipeline.py` first to extract the dataset, train the "
            "classifier, and generate results/engagement_pipeline_results.json."
        )
    else:
        eng_results = json.loads(eng_results_path.read_text())
        m1, m2, m3 = st.columns(3)
        m1.metric("6-class accuracy", f"{eng_results['accuracy_6class']:.1%}")
        m2.metric("Binary (Engaged vs. Not) accuracy", f"{eng_results['accuracy_macro_binary']:.1%}")
        m3.metric("n images", eng_results["n_images"])

        @st.cache_resource
        def get_engagement_model():
            image_paths, labels = [], []
            for img_path in EXTRACT_DIR.rglob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = normalize_label(img_path.parent.name)
                if label not in MACRO_CATEGORY:
                    continue
                image_paths.append(img_path)
                labels.append(label)
            X = np.stack([featurize(p) for p in image_paths])
            y = np.array(labels)
            model = CatBoostClassifier(iterations=80, depth=5, learning_rate=0.1,
                                        loss_function="MultiClass", random_seed=42,
                                        verbose=False, thread_count=2)
            model.fit(X, y)
            return model, image_paths, labels

        with st.spinner("Loading engagement classifier (cached after first load)..."):
            eng_model, eng_paths, eng_labels = get_engagement_model()

        st.divider()
        st.markdown("**Simulate a classroom session**")
        st.caption(
            "In the real design, this is the only output a mentor or faculty member would "
            "ever see: a single aggregate statistic, never a per-frame or per-identity result. "
            "The image grid below is shown here just to illustrate the mechanism, since these "
            "are public dataset photos, not real students under actual deployment."
        )
        session_size = st.slider("Simulated session size (frames)", 10, 60, 24)
        if st.button("🔄 Sample a new session"):
            st.session_state["eng_seed"] = np.random.randint(0, 100000)
        seed = st.session_state.get("eng_seed", 42)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eng_paths), size=session_size, replace=False)

        sample_feats = np.stack([featurize(eng_paths[i]) for i in idx])
        preds = eng_model.predict(sample_feats).flatten()
        macro_preds = [MACRO_CATEGORY[p] for p in preds]
        pct_engaged = float(np.mean([m == "Engaged" for m in macro_preds]))

        st.metric("Aggregate engagement for this simulated session", f"{pct_engaged:.1%}")

        n_show = min(12, session_size)
        grid_cols = st.columns(6)
        for j in range(n_show):
            img = Image.open(eng_paths[idx[j]]).resize((100, 100))
            with grid_cols[j % 6]:
                st.image(img, caption=f"pred: {preds[j]}", use_container_width=True)
        if session_size > n_show:
            st.caption(f"(+{session_size - n_show} more frames included in the aggregate, not shown)")
