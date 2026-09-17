# VIT-EAR: Explainable, Cost-Sensitive Academic-Risk Prediction

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green)

VIT-EAR (Explainable, Adaptive, cost-sensitive academic-Risk framework) is a machine
learning prototype for early prediction of academic risk and dropout at Indian
engineering colleges, built around VIT Vellore as the reference institution. It's a
proof of concept: every modeling and explainability piece is implemented and tested
end to end, using real public data where a suitable dataset exists and a clearly
labeled synthetic dataset (built to a real VIT-style schema) everywhere else.

A review of 22 recent papers on student dropout prediction turned up the same
weaknesses over and over: models trained on Western data that don't generalize to
other institutions, class imbalance handled as a resampling problem instead of a real
cost problem, SHAP/LIME explanations that never turn into anything a mentor can act
on, and outcomes treated as a flat binary dropout flag when institutions actually
track a graded severity ladder. VIT-EAR tries to address these directly.

## What's different here compared to a standard dropout classifier

- **Ordinal risk levels instead of a binary flag**: predicts one of four severity
  levels mapped to real institutional consequences: `0` No Risk, `1`
  Backlog / Academic-Probation Risk, `2` Attendance-Debarment Risk, `3` Withdrawal /
  Discontinuation Risk
- **A cost matrix that treats mistakes asymmetrically**: penalizes under-predicting
  severity 3x more heavily than over-predicting it, and the penalty grows with how
  far off the prediction is (see the formula below)
- **A deterministic override for things that aren't actually probabilities**: if a
  student's maximum reachable attendance already falls below the debarment
  threshold, that's a certainty, not something a classifier needs to guess at, so a
  rule layered on top of the model floors the prediction regardless of what the
  model itself says
- **Explanations that point to an action, not just a feature list**: SHAP
  contributions get grouped into 7 source clusters, and whichever one contributes
  most becomes a recommended intervention type a mentor can act on
- **A working camera-engagement pipeline with privacy built into the design**: a
  6-class classifier trained on real image data, with only an aggregate engagement
  percentage ever surfaced, never a per-frame or per-identity result
- **An interactive dashboard, not just notebooks**: pick a student and compare both
  models' predictions with live SHAP charts, or build a hypothetical student with
  sliders and watch the prediction change

The cost matrix:

```
C(y, ŷ) = w_under * max(0, y − ŷ)^p + w_over * max(0, ŷ − y)^p     (w_under=3, w_over=1, p=1)
```

implemented with instance re-weighting during training and a Bayes-risk decision
rule at inference (`argmin_k Σⱼ P(y=j|x)·C(j,k)`) instead of the usual arg-max. The
deterministic override is `apply_deterministic_overrides` in
`src/cost_sensitive_model.py`.

## Project structure

```
implementation/
├── app.py                        # Streamlit dashboard (4 tabs)
├── requirements.txt
├── src/
│   ├── synthetic_data.py         # builds the India/VIT-style synthetic dataset
│   ├── download_baseline_data.py # fetches the real UCI benchmark dataset
│   ├── baseline_model.py         # literature-benchmark baseline (real data)
│   ├── cost_sensitive_model.py   # the actual VIT-EAR model
│   ├── cross_validation.py       # k-fold CV + Optuna hyperparameter search
│   └── engagement_pipeline.py    # camera-engagement classifier (real data)
├── data/
│   ├── synthetic/                # generated synthetic dataset (small, committed)
│   └── raw/                      # downloaded datasets (gitignored, see Setup)
└── results/                      # JSON/CSV outputs from each script (committed)
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate       # Windows Git Bash; use venv\Scripts\activate.bat for cmd.exe
python -m pip install -r requirements.txt
```

`data/raw/` isn't committed since it holds downloaded third-party datasets. Fetch
them with the scripts below before running the baseline or the engagement pipeline.

## Running it

Run these in order, since later scripts depend on earlier ones' output.

1. Generate the synthetic dataset:
   ```bash
   python src/synthetic_data.py --n 600 --seed 42
   ```
   Produces `data/synthetic/vit_ear_synthetic_v1.csv` plus a `.meta.json` file
   documenting which fields are real vs. fabricated, and two items still pending
   verification against VIT's actual policies (attendance-debarment thresholds and
   mentor-log category names).

2. Download the real baseline dataset:
   ```bash
   python src/download_baseline_data.py
   ```
   Fetches the UCI "Predict Students' Dropout and Academic Success" dataset
   (Portugal) into `data/raw/`. Used only for a literature-comparison baseline, kept
   separate from the VIT-EAR model itself.

3. Train the literature-benchmark baseline:
   ```bash
   python src/baseline_model.py
   ```
   CatBoost + SMOTE on the real UCI data, matching the setup in Jain et al. (2025)
   and Arevalo-Cordovilla & Pena (2025). Results go to
   `results/baseline_literature_benchmark.json`.

4. Train the proposed VIT-EAR model:
   ```bash
   python src/cost_sensitive_model.py
   ```
   Trains the ordinal risk classifier on the synthetic dataset: a plain in-house
   baseline for comparison, the cost-sensitive version (instance re-weighting +
   Bayes-risk decision rule + the deterministic override), and SHAP-based
   source-cluster routing, validated against the synthetic data's known ground
   truth. Results go to `results/cost_sensitive_model_results.json` and
   `results/test_predictions_with_routing.csv`.

5. Cross-validate and tune hyperparameters:
   ```bash
   python src/cross_validation.py
   ```
   5-fold CV + Optuna search on both the baseline and the proposed model, so the
   numbers below aren't just a lucky split. Results go to
   `results/cross_validation_results.json`.

6. Train and validate the engagement pipeline:
   ```bash
   python src/engagement_pipeline.py
   ```
   Download the dataset zip from
   [kaggle.com/datasets/deepramazumder/student-engagement-dataset](https://www.kaggle.com/datasets/deepramazumder/student-engagement-dataset)
   (CC0, 2,120 real images) to `data/raw/student-engagement-dataset.zip` first.
   Used as a substitute for DAiSEE, which needs a manual license-agreement request
   that couldn't be arranged in time. Trains a 6-class engagement classifier and
   demonstrates the privacy-preserving output design. Results go to
   `results/engagement_pipeline_results.json`.

## Dashboard

```bash
streamlit run app.py
```

Four tabs:
- **Student Explorer**: pick a synthetic student, compare both models' predictions
  and SHAP source-cluster charts, with a banner when the deterministic attendance
  override kicks in
- **What-If Simulator**: sliders (including semester progress) to build a
  hypothetical student and watch predictions update live
- **Model Performance**: pulls the headline numbers straight from `results/*.json`
- **Engagement Pipeline**: simulate a classroom session and see only the aggregate
  engagement statistic, plus a sample frame grid for illustration

First load trains a smaller model (150 iterations instead of 300) just for
responsiveness. The numbers reported below come from the full 300-iteration
standalone scripts, not the dashboard's own quick-trained copy.

## Results

### Literature-benchmark baseline (real UCI data, n=3,630)

| | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Single split | 0.930 | 0.910 | 0.971 |
| 5-fold CV (default params) | 0.912 ± 0.008 | 0.883 ± 0.011 | 0.956 ± 0.008 |
| 5-fold CV (Optuna-tuned) | 0.913 ± 0.008 | 0.884 ± 0.011 | 0.955 ± 0.008 |

Tuning barely moved this number, which is itself worth noting: the default settings
were already close to optimal on this dataset.

### VIT-EAR cost-sensitive model (synthetic data, n=600)

| Model | Accuracy | Macro F1 | Mean expected cost |
|---|---|---|---|
| Plain (arg-max), single split | 0.813 | 0.346 | 0.693 |
| Cost-sensitive (Bayes-risk), single split | 0.673 | 0.346 | 0.607 |
| Cost-sensitive, 5-fold CV (default params) | 0.643 ± 0.037 | 0.412 ± 0.065 | 0.603 ± 0.064 |
| Cost-sensitive, 5-fold CV (Optuna-tuned) | 0.713 ± 0.029 | 0.418 ± 0.064 | 0.558 ± 0.066 |

The cost-sensitive model gives up some raw accuracy in exchange for a lower expected
cost. That's the intended effect of the cost matrix, not a side effect. Unlike the
baseline, tuning here made a real difference (mean cost 0.603 to 0.558, accuracy
0.643 to 0.713 at the same time).

Per-class recall on Level 3 / Withdrawal isn't reported in the table above. There
were only 4 such students in that single test split, which is too small a sample to
say anything reliable, so the CV rows are the numbers to trust instead.

SHAP source-cluster routing recovered the correct synthetic driver cluster in 43.0%
of at-risk cases (a random guess across 7 clusters would get 14%), and matched the
correct intervention type in 43.8% of cases (a random guess across 4 types would get
25%). These figures are pooled across 5-fold cross-validation (121 at-risk students
total) rather than taken from one split, since a single split only has about 30
at-risk students, too few to measure a 7-way classification accuracy reliably. That
same measurement moved between 13% and 50% across a single split during development
purely from sample noise, which is why it's measured this way now. See
`cv_routing_validation` in `src/cost_sensitive_model.py`.

### Camera-engagement pipeline (real data, n=2,120)

| | Accuracy |
|---|---|
| 6-class engagement-state | 0.958 |
| Binary macro-category (Engaged vs. Not Engaged) | 1.000 |

Most of the confusion was between Bored and Drowsy, two states that look similar.
A simulated 40-frame session correctly surfaced only the aggregate statistic (47.5%
sustained engagement), never a per-frame or per-identity result.

## A design detail worth mentioning

Attendance risk started out as a static snapshot (mean/min attendance %) with no
sense of how much of the semester was left, so a student at 40% attendance in week 2
and the same percentage in week 12 looked identical to the model, even though those
are very different situations. `max_possible_attendance_pct` (computed from classes
conducted so far vs. remaining) fixes that as a single feature, feeding into the
deterministic override described above. An earlier version split this signal across
three separate correlated fields (classes conducted, classes remaining, and a
redundant certainty flag), which inflated the attendance cluster's SHAP-based
importance without adding any real signal. It's now one non-redundant feature, and
all seven source clusters end up with comparable feature counts (2-9 each) so the
SHAP-cluster routing stays fair.

## On the data

Everything in `data/synthetic/` is fabricated, built to demonstrate that the
modeling pipeline works end to end. It's not a measurement of real students at any
institution. Each script's docstring documents what's real vs. simulated at that
step, and the `.meta.json` next to the synthetic dataset gives field-by-field
provenance. The literature-benchmark baseline and the engagement-recognition
pipeline run on real public data; the VIT-EAR model itself runs on the synthetic
dataset because collecting real student data needs institutional ethics approval
this phase of the project doesn't have yet.

## Status and next steps

The proof-of-concept phase is done: all five pieces above (baseline, synthetic
data, engagement pipeline, cost-sensitive model, cross-validation) run end to end,
along with the dashboard. What's left isn't technical, it's institutional:
replacing the synthetic tier with real, consented data from a pilot at VIT (which
needs ethics approval and access to systems the project doesn't have yet), then
proper validation on that real data: ablation studies, fairness auditing across
demographic groups, robustness testing, and a decision on whether the contribution
is more suited to publication or a patent.

## License

MIT, see [LICENSE](LICENSE).

## Author

Guru Charan S ([@GuruSGC](https://github.com/GuruSGC)). Built as part of a
Foundations of Data Science course project at VIT Vellore.
