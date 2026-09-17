"""
Generates a synthetic student dataset for VIT-EAR, following the field schema a real
VIT dataset would use. All rows here are fabricated - the goal is to have something
that exercises the full pipeline (ordinal risk model, intervention routing, SHAP
explanations) before real student data is available.

Since we generate the data, we also know which factor actually drove each student's
simulated risk. That's stored in `true_dominant_cluster` and used later to check
whether the SHAP routing logic recovers the right cluster - a check that's only
possible because we know the ground truth here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

SOURCE_CLUSTERS = [
    "academic",
    "attendance",
    "pedagogy",
    "diagnostic",
    "visual_engagement",
    "behavioural",
    "financial",
]

RISK_LEVELS = {
    0: "No Risk",
    1: "Backlog / Academic-Probation Risk",
    2: "Attendance-Debarment Risk",
    3: "Withdrawal / Discontinuation Risk",
}

INTERVENTION_TYPES = ["none", "academic", "pedagogical", "psychosocial", "financial"]

BRANCHES = ["SCOPE-CSE", "SENSE-ECE", "SMEC-MECH", "SCE-CIVIL", "SBST-BIOTECH"]
HOME_STATES = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Maharashtra",
               "Delhi-NCR", "Uttar Pradesh", "West Bengal", "Bihar", "Other"]
SCHOOLING_MEDIUM = ["State Board", "CBSE", "ICSE", "International"]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate(n_students: int, weeks: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for i in range(n_students):
        student_id = f"S{i:05d}"  # synthetic hashed-style id, not a real VIT registration number

        # Academic and demographic fields
        entrance_percentile = _clip(rng.normal(75, 15), 5, 99.9)
        cgpa_base = _clip(3 + (entrance_percentile / 100) * 6 + rng.normal(0, 0.8), 0, 10)
        backlogs_count = int(_clip(rng.poisson(lam=max(0.1, (8 - cgpa_base))), 0, 8))
        backlogs_age_max_weeks = 0 if backlogs_count == 0 else int(_clip(rng.exponential(10), 1, 60))
        branch = rng.choice(BRANCHES)
        admission_category = rng.choice(["Government Quota", "Management Quota"], p=[0.6, 0.4])
        social_category = rng.choice(["General", "OBC", "SC", "ST", "EWS"], p=[0.45, 0.27, 0.10, 0.05, 0.13])
        home_state = rng.choice(HOME_STATES)
        schooling_medium = rng.choice(SCHOOLING_MEDIUM, p=[0.35, 0.45, 0.15, 0.05])
        hosteller = rng.choice(["Hosteller", "DayScholar"], p=[0.7, 0.3])

        # Attendance-pattern features
        # debarment/condonation thresholds below are placeholders - need to check
        # the actual current values in the student handbook
        attendance_debarment_threshold = 65.0
        attendance_condonation_band_low = 65.0
        attendance_condonation_band_high = 75.0
        mean_attendance = _clip(rng.normal(80 - backlogs_count * 2, 12), 30, 100)
        min_attendance = _clip(mean_attendance - rng.exponential(8), 20, 100)
        consecutive_absence_streak_max = int(_clip(rng.exponential(3) + (0 if min_attendance > 70 else 4), 0, 21))
        attendance_trend = rng.choice(["declining", "stable", "recovering"],
                                       p=[0.3, 0.55, 0.15] if mean_attendance < 75 else [0.1, 0.75, 0.15])
        timing_of_decline = (rng.choice(["early", "late"]) if attendance_trend == "declining" else "none")

        # how far into the semester we are, and whether the debarment threshold is
        # still reachable with perfect attendance from here. this is a fact we can
        # compute directly, not something the model needs to guess at.
        # using 50 as a rough average session count per semester for now
        classes_total_in_sem = 50
        semester_progress_frac = float(_clip(rng.uniform(0.1, 0.95), 0.1, 0.95))
        classes_conducted_so_far = max(1, round(semester_progress_frac * classes_total_in_sem))
        classes_remaining_in_sem = classes_total_in_sem - classes_conducted_so_far
        classes_attended_so_far = round(mean_attendance / 100 * classes_conducted_so_far)
        max_possible_attendance_pct = (
            (classes_attended_so_far + classes_remaining_in_sem) / classes_total_in_sem * 100
        )
        attendance_debarment_mathematically_certain = int(
            max_possible_attendance_pct < attendance_debarment_threshold
        )

        # Pedagogy feedback fields
        avg_pace_score = _clip(rng.normal(0, 0.8), -1, 1)  # -1 too slow/fast, 0 about right
        avg_clarity_score = _clip(rng.normal(3.5, 1.0), 1, 5)
        has_doubt_events = rng.random() < 0.8
        avg_doubt_resolution_score = _clip(rng.normal(3.3, 1.1), 1, 5) if has_doubt_events else np.nan
        avg_teaching_style_fit_score = _clip(rng.normal(3.4, 1.0), 1, 5)

        # Diagnostic micro-assessment scores
        diagnostic_content_score = _clip((cgpa_base / 10) + rng.normal(0, 0.15), 0, 1)
        diagnostic_delivery_sensitive_score = _clip(
            diagnostic_content_score + (avg_clarity_score - 3) * 0.05 + rng.normal(0, 0.1), 0, 1
        )

        # Camera engagement signal. In the real pipeline (engagement_pipeline.py) this
        # comes from an actual image classifier trained on real data; here we just
        # fake a per-student average exposure value directly since that pipeline runs
        # separately from this generator.
        avg_session_engagement_exposure = _clip(
            0.5 + (mean_attendance - 80) / 100 + rng.normal(0, 0.12), 0, 1
        )

        # Institutional behavioural logs
        vtop_login_freq_per_week = _clip(rng.normal(10 - backlogs_count * 0.5, 3), 0, 30)
        submission_timing_relative = _clip(rng.normal(0, 1) + (0.3 if backlogs_count > 2 else -0.1), -3, 3)
        footfall_library_freq_per_week = _clip(rng.normal(3, 2), 0, 20)
        if hosteller == "Hosteller":
            hostel_curfew_violations = int(_clip(rng.poisson(1 if mean_attendance > 70 else 3), 0, 20))
        else:
            hostel_curfew_violations = np.nan  # day scholars don't have curfew data, don't fill with 0

        # Fee payment and scholarship status
        scholarship_category = rng.choice(
            ["No Scholarship", "Merit-Based", "Category-Based", "Sports-Cultural Quota", "Institutional Need-Based"],
            p=[0.55, 0.15, 0.15, 0.05, 0.10],
        )
        fee_stress = rng.random() < (0.25 if scholarship_category == "No Scholarship" else 0.10)
        fee_payment_status = (
            rng.choice(["On Installment (Overdue)", "Fee Due >30 Days"], p=[0.6, 0.4])
            if fee_stress else
            rng.choice(["Fully Paid", "On Installment (On-Time)"], p=[0.7, 0.3])
        )
        education_loan_status = rng.choice(
            ["No Loan", "Active Loan (No Stress Flagged)", "Active Loan (Repayment Stress Flagged)"],
            p=[0.55, 0.30, 0.15] if not fee_stress else [0.35, 0.30, 0.35],
        )
        family_income_bracket = rng.choice(["<2.5 LPA", "2.5-8 LPA", ">8 LPA"], p=[0.35, 0.45, 0.20])
        first_gen_college_goer = rng.choice(["Yes", "No"], p=[0.4, 0.6])

        # build the latent risk score - we know these numbers so we can check
        # the routing logic against ground truth later
        cluster_scores = {
            "academic": max(0.0, (5.5 - cgpa_base) / 5.5) * 0.6 + (backlogs_count / 8) * 0.4,
            "attendance": max(0.0, (75 - mean_attendance) / 75) * 0.5
                          + (consecutive_absence_streak_max / 21) * 0.2
                          + attendance_debarment_mathematically_certain * 0.3,
            "pedagogy": max(0.0, (3 - avg_teaching_style_fit_score) / 3) * 0.5
                        + max(0.0, (3 - avg_clarity_score) / 3) * 0.5,
            "diagnostic": max(0.0, (0.5 - diagnostic_content_score) / 0.5),
            "visual_engagement": max(0.0, (0.5 - avg_session_engagement_exposure) / 0.5),
            "behavioural": max(0.0, -submission_timing_relative / 3) * 0.5
                           + (0 if np.isnan(hostel_curfew_violations) else hostel_curfew_violations / 20) * 0.5,
            "financial": (0.6 if fee_stress else 0.0)
                         + (0.4 if education_loan_status.endswith("Stress Flagged)") else 0.0),
        }
        # dominant cluster gets picked later, after normalizing scores across the
        # population - the raw scale differs a lot between clusters (financial is
        # basically a step function, academic is continuous) so picking it here
        # would just favor whichever cluster has the widest range
        latent_risk_score = (
            0.20 * cluster_scores["academic"]
            + 0.20 * cluster_scores["attendance"]
            + 0.12 * cluster_scores["pedagogy"]
            + 0.12 * cluster_scores["diagnostic"]
            + 0.10 * cluster_scores["visual_engagement"]
            + 0.11 * cluster_scores["behavioural"]
            + 0.15 * cluster_scores["financial"]
            + rng.normal(0, 0.06)
        )
        latent_risk_score = float(_clip(latent_risk_score, 0, 1.2))

        # risk_level gets bucketed after everyone is generated, using quantiles on
        # latent_risk_score (see below) so we control the class balance directly
        risk_level = None

        rows.append(dict(
            student_id=student_id,
            cgpa_cumulative=round(cgpa_base, 2),
            backlogs_count=backlogs_count,
            backlogs_age_max_weeks=backlogs_age_max_weeks,
            branch=branch,
            entrance_percentile=round(entrance_percentile, 1),
            admission_category=admission_category,
            social_category=social_category,  # ETHICAL-USE-ONLY: fairness eval only, never a model input
            home_state=home_state,
            schooling_medium=schooling_medium,
            hosteller_flag=hosteller,
            mean_attendance_pct=round(mean_attendance, 1),
            min_attendance_pct=round(min_attendance, 1),
            attendance_debarment_threshold=attendance_debarment_threshold,
            attendance_condonation_band_low=attendance_condonation_band_low,
            attendance_condonation_band_high=attendance_condonation_band_high,
            consecutive_absence_streak_max=consecutive_absence_streak_max,
            attendance_trend=attendance_trend,
            timing_of_decline=timing_of_decline,
            classes_total_in_sem=classes_total_in_sem,
            classes_conducted_so_far=classes_conducted_so_far,
            classes_remaining_in_sem=classes_remaining_in_sem,
            max_possible_attendance_pct=round(max_possible_attendance_pct, 1),
            attendance_debarment_mathematically_certain=attendance_debarment_mathematically_certain,
            avg_pace_score=round(avg_pace_score, 2),
            avg_clarity_score=round(avg_clarity_score, 2),
            avg_doubt_resolution_score=(round(avg_doubt_resolution_score, 2)
                                         if not np.isnan(avg_doubt_resolution_score) else np.nan),
            avg_teaching_style_fit_score=round(avg_teaching_style_fit_score, 2),
            diagnostic_content_score=round(diagnostic_content_score, 3),
            diagnostic_delivery_sensitive_score=round(diagnostic_delivery_sensitive_score, 3),
            avg_session_engagement_exposure=round(avg_session_engagement_exposure, 3),
            vtop_login_freq_per_week=round(vtop_login_freq_per_week, 2),
            submission_timing_relative=round(submission_timing_relative, 2),
            footfall_library_freq_per_week=round(footfall_library_freq_per_week, 2),
            hostel_curfew_violations=hostel_curfew_violations,  # NaN = not applicable (day scholar)
            fee_payment_status=fee_payment_status,
            scholarship_category=scholarship_category,
            education_loan_status=education_loan_status,
            family_income_bracket=family_income_bracket,
            first_gen_college_goer=first_gen_college_goer,
            latent_risk_score=round(latent_risk_score, 4),
            **{f"_cs_{k}": v for k, v in cluster_scores.items()},  # raw cluster scores, dropped before saving
        ))

    df = pd.DataFrame(rows)

    # pick the dominant driver cluster from normalized scores, not raw ones - min-max
    # scale each cluster's column across the population first so a cluster with a wide
    # natural range doesn't automatically win just because of its scale
    cs_cols = [c for c in df.columns if c.startswith("_cs_")]
    norm = df[cs_cols].copy()
    for c in cs_cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - lo) / (hi - lo) if hi > lo else 0.0
    df["true_dominant_cluster"] = norm.idxmax(axis=1).str.replace("_cs_", "", regex=False)
    df = df.drop(columns=cs_cols)

    # bucket into the 4 risk levels using quantiles, targeting roughly an
    # 80/12/5/3 split so every level actually shows up in the data
    q1, q2, q3 = df["latent_risk_score"].quantile([0.80, 0.92, 0.97])
    df["risk_level"] = np.select(
        [df["latent_risk_score"] < q1, df["latent_risk_score"] < q2, df["latent_risk_score"] < q3],
        [0, 1, 2],
        default=3,
    )

    # if a student can't reach the debarment threshold even with perfect attendance,
    # that's a certainty and shouldn't depend on whatever the quantile bucketing gave
    # them - force it to at least Level 2, and mark attendance as the driver since
    # that's the actual reason
    certain_mask = df["attendance_debarment_mathematically_certain"] == 1
    df.loc[certain_mask & (df["risk_level"] < 2), "risk_level"] = 2
    df.loc[certain_mask, "true_dominant_cluster"] = "attendance"

    df["risk_level_label"] = df["risk_level"].map(RISK_LEVELS)

    # Mentor-log weak label: Level 0 students get "none"; everyone else is routed to the
    # intervention type implied by their (normalized) dominant synthetic driver cluster.
    cluster_to_intervention = {
        "academic": "academic", "attendance": "academic", "pedagogy": "pedagogical",
        "diagnostic": "pedagogical", "visual_engagement": "pedagogical",
        "behavioural": "psychosocial", "financial": "financial",
    }
    df["mentor_flag_category"] = np.where(
        df["risk_level"] == 0, "none", df["true_dominant_cluster"].map(cluster_to_intervention)
    )

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="number of synthetic students")
    ap.add_argument("--weeks", type=int, default=1, help="reserved for future weekly-resolution extension")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--out", type=str, default="data/synthetic/vit_ear_synthetic_v1.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate(args.n, args.weeks, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta = {
        "provenance": "Synthetic data generated for pipeline testing. Not real student records.",
        "n_students": args.n,
        "seed": args.seed,
        "risk_level_distribution": df["risk_level"].value_counts().sort_index().to_dict(),
        "mentor_flag_distribution": df["mentor_flag_category"].value_counts().to_dict(),
        "placeholders_pending_verification": [
            "attendance_debarment_threshold and condonation band need checking against the current student handbook",
            "mentor_flag_category set needs checking against the real mentor-mentee log categories",
        ],
        "ethical_use_only_fields": ["social_category (used for fairness evaluation only, never a predictive input)"],
        "not_applicable_semantics": ["hostel_curfew_violations is NaN for day scholars, not zero"],
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(df)} synthetic student rows to {out_path}")
    print(f"Wrote metadata to {meta_path}")
    print("Risk level distribution:", meta["risk_level_distribution"])
    print("Mentor flag distribution:", meta["mentor_flag_distribution"])


if __name__ == "__main__":
    main()
