"""
Camera-engagement classifier, trained and validated on a real public dataset before
its output is trusted as a model input.

DAiSEE, the dataset originally planned for this, needs a manual license-agreement
request to its authors that couldn't be arranged in time. Using the
"Student-Engagement-Dataset" on Kaggle instead (deepramazumder/student-engagement-dataset,
CC0, about 2120 real images across 6 states). These are static images rather than
video, so there's no temporal signal within a single frame.

Folder structure (per the Kaggle dataset card):
  Engaged/            (engaged in some form)
    Confused/
    Focused/
    Frustrated/
  Not Engaged/         (disengaged)
    Bored/
    Drowsy/
    Looking Away/

What this script does:
  1. Loads and featurizes images as small grayscale thumbnails - simple classical
     features rather than a deep CNN, mainly for speed.
  2. Trains a 6-class CatBoost classifier and evaluates both the 6-class accuracy
     and the derived binary Engaged vs. Not Engaged accuracy.
  3. Demonstrates the privacy-preserving design the project is built around:
     simulates a "session" as a batch of frames, classifies each one, and outputs
     only the aggregate percentage engaged for that session - never a per-frame
     or per-identity result.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RAW_ZIP = Path("data/raw/student-engagement-dataset.zip")
EXTRACT_DIR = Path("data/raw/student_engagement")
THUMB_SIZE = (32, 32)  # small on purpose, classical features instead of a deep CNN, for speed

MACRO_CATEGORY = {
    "confused": "Engaged", "focused": "Engaged", "frustrated": "Engaged",
    "bored": "Not Engaged", "drowsy": "Not Engaged", "lookingaway": "Not Engaged",
}


def ensure_extracted():
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.jpg")):
        return
    if not RAW_ZIP.exists():
        raise FileNotFoundError(
            f"{RAW_ZIP} not found. Download it from "
            "https://www.kaggle.com/datasets/deepramazumder/student-engagement-dataset "
            f"and save it to {RAW_ZIP} before running this script."
        )
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {RAW_ZIP} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(RAW_ZIP) as z:
        z.extractall(EXTRACT_DIR)


def normalize_label(folder_name: str) -> str:
    return folder_name.lower().replace(" ", "").replace("-", "").replace("_", "")


def load_dataset():
    ensure_extracted()
    image_paths, labels = [], []
    for img_path in EXTRACT_DIR.rglob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = normalize_label(img_path.parent.name)
        if label not in MACRO_CATEGORY:
            continue  # skip the top-level Engaged/Not Engaged grouping folders themselves
        image_paths.append(img_path)
        labels.append(label)
    if not image_paths:
        raise RuntimeError(
            f"No labeled images found under {EXTRACT_DIR}. Check the extracted folder "
            "structure matches the expected sub-state folder names."
        )
    print(f"Found {len(image_paths)} labeled images across classes: "
          f"{sorted(set(labels))}")
    return image_paths, labels


def featurize(img_path: Path) -> np.ndarray:
    """A small grayscale thumbnail, flattened. Not a deep CNN - chosen for speed,
    since the goal is validating the pipeline (image -> engagement label -> session
    aggregate) rather than chasing state-of-the-art image classification accuracy."""
    img = Image.open(img_path).convert("L").resize(THUMB_SIZE)
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def main():
    image_paths, labels = load_dataset()
    print("Featurizing images (small grayscale thumbnails)...")
    X = np.stack([featurize(p) for p in image_paths])
    y = np.array(labels)
    y_macro = np.array([MACRO_CATEGORY[l] for l in labels])

    X_train, X_test, y_train, y_test, y_macro_train, y_macro_test = train_test_split(
        X, y, y_macro, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=150, depth=6, learning_rate=0.08, loss_function="MultiClass",
        random_seed=42, verbose=False, thread_count=2,
    )
    print(f"Training 6-class engagement-state classifier on {len(X_train)} images...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test).flatten()
    acc_6class = accuracy_score(y_test, pred)
    pred_macro = np.array([MACRO_CATEGORY[p] for p in pred])
    acc_macro = accuracy_score(y_macro_test, pred_macro)

    print(f"\n6-class accuracy: {acc_6class:.4f}")
    print(f"Derived binary (Engaged vs. Not Engaged) accuracy: {acc_macro:.4f}")
    print("\nClassification report (6-class):")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (macro-category, [ [TN FP] [FN TP] ] with 'Engaged' as positive):")
    print(confusion_matrix(y_macro_test, pred_macro, labels=["Not Engaged", "Engaged"]))

    # simulate a session-level aggregate: sample a batch of held-out test images to
    # stand in for frames from one lecture, classify each, and output only the
    # aggregate percentage engaged - never a per-frame or per-identity result
    rng = np.random.default_rng(7)
    session_size = 40
    session_idx = rng.choice(len(X_test), size=session_size, replace=False)
    session_pred = model.predict(X_test[session_idx]).flatten()
    session_macro = [MACRO_CATEGORY[p] for p in session_pred]
    pct_engaged = float(np.mean([m == "Engaged" for m in session_macro]))
    print(f"\nSimulated session-level aggregate (n={session_size} frames): "
          f"{pct_engaged:.1%} sustained engagement")

    Path("results").mkdir(exist_ok=True)
    results = {
        "provenance": "Real public data (Kaggle deepramazumder/student-engagement-dataset, CC0), "
                       "used as a substitute for DAiSEE. Static images, not video, so there's no "
                       "temporal signal within a single image.",
        "n_images": len(image_paths),
        "classes": sorted(set(labels)),
        "accuracy_6class": acc_6class,
        "accuracy_macro_binary": acc_macro,
        "simulated_session_aggregate_pct_engaged": pct_engaged,
        "simulated_session_size": session_size,
        "note": "The simulated session aggregate uses held-out test images standing in for "
                "frames from one lecture. It's a demo of the aggregate-only output design "
                "(never per-frame or per-identity), not a claim about a real classroom.",
    }
    with open("results/engagement_pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/engagement_pipeline_results.json")


if __name__ == "__main__":
    main()
