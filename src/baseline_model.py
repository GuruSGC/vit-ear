"""
Reproduces a literature-comparable baseline on the real UCI dropout dataset, matching
the setup used in Jain et al. (2025) and Arevalo-Cordovilla & Pena (2025): CatBoost +
SMOTE, default 0.5 threshold, standard accuracy/F1/AUC.

This is not a VIT-EAR result - it's a separate reference point on real data, kept
apart from the actual proposed model in cost_sensitive_model.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


def load_data():
    df = pd.read_csv("data/raw/uci_dropout_academic_success.csv")
    # drop "Enrolled" since that's still in progress, not a resolved outcome - most
    # of the reviewed papers just compare Dropout vs Graduate
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    y = (df["Target"] == "Dropout").astype(int)
    X = df.drop(columns=["Target"])
    return X, y


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} rows after restricting to Dropout/Graduate. "
          f"Dropout rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE on the training split only (never on test data, to avoid leakage)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_res)} training rows, "
          f"class balance {y_train_res.mean():.3f}")

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="F1",
        random_seed=42, verbose=False, thread_count=4,
    )
    model.fit(X_train_res, y_train_res)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)  # default threshold

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
    }

    print("\n=== baseline results (UCI dataset, Dropout vs Graduate) ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["Graduate", "Dropout"]))

    Path("results").mkdir(exist_ok=True)
    with open("results/baseline_literature_benchmark.json", "w") as f:
        json.dump({
            "provenance": "Real public data (UCI, Portugal). Literature comparison only, not a VIT-EAR result.",
            "n_train_after_smote": int(len(X_train_res)),
            "n_test": int(len(X_test)),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }, f, indent=2)
    print("\nSaved results/baseline_literature_benchmark.json")


if __name__ == "__main__":
    main()
