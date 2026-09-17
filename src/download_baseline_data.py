"""
Downloads the real public dataset used for the literature-benchmark baseline
(matching the setup in Jain et al. 2025 / Arevalo-Cordovilla & Pena 2025). Kept
separate from the actual VIT-EAR model, which runs on the synthetic dataset from
synthetic_data.py.

Source: UCI Machine Learning Repository, dataset id 697,
"Predict Students' Dropout and Academic Success" (Portugal).
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching UCI dataset 697 (Predict Students' Dropout and Academic Success)...")
    dataset = fetch_ucirepo(id=697)

    X = dataset.data.features
    y = dataset.data.targets
    df = X.copy()
    df["Target"] = y.iloc[:, 0]

    out_path = out_dir / "uci_dropout_academic_success.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols to {out_path}")
    print("Target value counts:")
    print(df["Target"].value_counts())


if __name__ == "__main__":
    main()
