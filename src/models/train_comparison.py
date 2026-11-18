"""Train, evaluate, and compare 5 distinct ML models across algorithm families using GroupKFold."""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.features.build_features import FEATURE_COLUMNS
from src.models.metrics import compute_observation_metrics


def get_model_slate() -> Dict[str, Tuple[str, object, int]]:
    """Return 5 models from distinct algorithm families with configurations and fit sample limits."""
    return {
        "LogisticRegression": (
            "Linear/Probabilistic",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, random_state=42, class_weight="balanced", solver="lbfgs")),
            ]),
            300000,
        ),
        "WeightedKNN": (
            "Instance-based",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance", algorithm="kd_tree", n_jobs=-1)),
            ]),
            50000,
        ),
        "RandomForest": (
            "Bagged Tree Ensemble",
            RandomForestClassifier(
                n_estimators=60,
                max_depth=12,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
            200000,
        ),
        "HistGradientBoosting": (
            "Gradient Boosting",
            HistGradientBoostingClassifier(
                max_iter=80,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                class_weight="balanced",
            ),
            400000,
        ),
        "MLPClassifier": (
            "Neural Network",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=25,
                    early_stopping=True,
                    random_state=42,
                )),
            ]),
            200000,
        ),
    }


def run_grouped_experiments(
    train_features_path: str = "artifacts/results/train_features.csv.gz",
    test_features_path: str = "artifacts/results/student_test_features.csv.gz",
    artifacts_dir: str = "artifacts/models",
    results_dir: str = "artifacts/results",
):
    """Perform GroupKFold CV on train trips and evaluate on student_test trips."""
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading feature tables from compressed archives...")
    train_df = pd.read_csv(train_features_path)
    test_df = pd.read_csv(test_features_path)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values
    groups_train = train_df["trip_id"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    print(f"Train samples: {len(train_df):,} rows from {train_df['trip_id'].nunique()} trips")
    print(f"Test samples:  {len(test_df):,} rows from {test_df['trip_id'].nunique()} trips")

    models = get_model_slate()
    comparison_records = []
    gkf = GroupKFold(n_splits=5)

    for model_name, (family, model, max_fit_samples) in models.items():
        print(f"\n========================================================")
        print(f"Evaluating {model_name} (Family: {family})")
        print(f"========================================================")

        # 1. Grouped Cross-Validation (3 folds for quick, robust validation)
        cv_macro_f1s = []
        cv_top1_accs = []
        cv_log_losses = []

        fold = 0
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
            fold += 1
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va = X_train[val_idx]
            df_va = train_df.iloc[val_idx].copy()

            if len(X_tr) > max_fit_samples:
                np.random.seed(42 + fold)
                sub_idx = np.random.choice(len(X_tr), max_fit_samples, replace=False)
                X_tr_fit, y_tr_fit = X_tr[sub_idx], y_tr[sub_idx]
            else:
                X_tr_fit, y_tr_fit = X_tr, y_tr

            t0 = time.time()
            model.fit(X_tr_fit, y_tr_fit)
            t_fit = time.time() - t0

            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_va)[:, 1]
            else:
                val_probs = model.decision_function(X_va)

            val_metrics = compute_observation_metrics(df_va, val_probs)
            cv_macro_f1s.append(val_metrics["cand_macro_f1"])
            cv_top1_accs.append(val_metrics["obs_top1_accuracy"])
            cv_log_losses.append(val_metrics["cand_log_loss"])
            print(f"  Fold {fold}: Macro F1 = {val_metrics['cand_macro_f1']:.4f}, Top-1 Acc = {val_metrics['obs_top1_accuracy']:.4f}, LogLoss = {val_metrics['cand_log_loss']:.4f} ({t_fit:.1f}s)")
            if fold >= 3:
                break

        mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))
        mean_cv_top1_acc = float(np.mean(cv_top1_accs))

        # 2. Refit on entire training partition
        print(f"  Fitting {model_name} on training partition...")
        if len(X_train) > max_fit_samples:
            np.random.seed(42)
            sub_idx = np.random.choice(len(X_train), max_fit_samples, replace=False)
            X_full_fit, y_full_fit = X_train[sub_idx], y_train[sub_idx]
        else:
            X_full_fit, y_full_fit = X_train, y_train

        t_start = time.time()
        model.fit(X_full_fit, y_full_fit)
        full_train_time = round(time.time() - t_start, 2)

        # 3. Evaluate on Student-Test partition
        t_infer_start = time.time()
        if hasattr(model, "predict_proba"):
            test_probs = model.predict_proba(X_test)[:, 1]
        else:
            test_probs = model.decision_function(X_test)
        total_infer_time = time.time() - t_infer_start
        latency_ms_per_obs = round((total_infer_time / test_df["obs_idx"].nunique()) * 1000, 2)

        test_metrics = compute_observation_metrics(test_df, test_probs)

        # Save model artifact
        artifact_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
        joblib.dump(model, artifact_path)

        record = {
            "Model": model_name,
            "Algorithm Family": family,
            "CV Macro F1-score": round(mean_cv_macro_f1, 4),
            "CV Top-1 Accuracy": round(mean_cv_top1_acc, 4),
            "Student-Test Top-1 Accuracy": test_metrics["obs_top1_accuracy"],
            "Student-Test Macro F1-score": test_metrics["cand_macro_f1"],
            "Student-Test Precision": test_metrics["cand_precision"],
            "Student-Test Recall": test_metrics["cand_recall"],
            "Student-Test Log Loss": test_metrics["cand_log_loss"],
            "Training Time (s)": full_train_time,
            "Inference Latency (ms/obs)": latency_ms_per_obs,
            "Artifact Path": artifact_path,
        }
        comparison_records.append(record)
        print(f"  Student-Test Results: Top-1 Acc = {test_metrics['obs_top1_accuracy']:.4f}, Macro F1 = {test_metrics['cand_macro_f1']:.4f}, LogLoss = {test_metrics['cand_log_loss']:.4f}")

    comp_df = pd.DataFrame(comparison_records)
    comp_csv = os.path.join(results_dir, "model_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved consolidated model comparison to {comp_csv}")

    try:
        print("\n" + comp_df.to_markdown(index=False))
    except Exception:
        print("\n" + comp_df.to_string(index=False))
    return comp_df


if __name__ == "__main__":
    run_grouped_experiments()
