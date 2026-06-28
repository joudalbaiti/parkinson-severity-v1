"""
Parkinson's Disease Severity Assessment — Model Training
Predicts MDS-UPDRS finger-tapping severity score (0–4) from extracted features.

Dataset: severity_dataset_dropped_correlated_columns.csv
         (place in the same directory before running)

Usage:
    python train.py
    python train.py --data path/to/your_features.csv
"""

import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr


def load_data(path: str):
    df = pd.read_csv(path)

    # Features: all columns between wrist movement and acceleration
    feature_cols = df.loc[:, "wrist_mvmnt_x_median":"acceleration_min_trimmed"].columns.tolist()
    X = df[feature_cols].values
    y = df["Rating"].values

    # Patient ID for leave-one-patient-out cross-validation
    def parse_patient_id(filename: str) -> str:
        if filename.startswith("NIH"):
            return filename.split("-")[0]
        parts = filename.split("-")
        return parts[-3]

    patient_ids = df["filename"].apply(parse_patient_id).values

    return X, y, patient_ids, feature_cols


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, _ = pearsonr(y_true, y_pred)
    return {
        "MAE":       round(mean_absolute_error(y_true, y_pred), 4),
        "R²":        round(r2_score(y_true, y_pred), 4),
        "Pearson r": round(r, 4),
    }


def train(data_path: str):
    print(f"Loading data from: {data_path}")
    X, y, patient_ids, feature_cols = load_data(data_path)
    print(f"  {X.shape[0]} samples | {X.shape[1]} features | {len(np.unique(patient_ids))} patients\n")

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    all_preds, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=patient_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_cols)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        all_preds.extend(preds)
        all_labels.extend(y_test)

        if (fold + 1) % 50 == 0:
            print(f"  Fold {fold + 1}/{logo.get_n_splits(groups=patient_ids)} done...")

    results = evaluate(np.array(all_labels), np.array(all_preds))

    print("\n── Results (Leave-One-Patient-Out CV) ──")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    print("────────────────────────────────────────")
    print("\nNote: The original paper (Islam et al., 2023) achieved MAE = 0.58.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parkinson severity model")
    parser.add_argument(
        "--data",
        default="severity_dataset_dropped_correlated_columns.csv",
        help="Path to features CSV file",
    )
    args = parser.parse_args()
    train(args.data)
