"""Bayesian Hyperparameter Optimization using Optuna on the best-performing model (HistGradientBoosting).

Runs >= 50 trials on 5-fold GroupKFold, maximizing Macro F1.
Saves Optuna study, plots, and a final Tuned vs Untuned comparison table.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score
from src.features.build_features import FEATURE_COLUMNS
from src.models.metrics import compute_observation_metrics

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Run: pip install optuna")

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAIN_FEATURES = "artifacts/results/train_features.csv.gz"
TEST_FEATURES  = "artifacts/results/student_test_features.csv.gz"
OUT_DIR        = "artifacts/bayesian_opt"
MODEL_DIR      = "artifacts/models"

N_TRIALS       = 50
N_CV_FOLDS     = 5


def objective(trial, X_train, y_train, groups_train):
    """Optuna objective: maximize mean Macro-F1 across GroupKFold folds."""
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter":         trial.suggest_int("max_iter", 50, 300),
        "max_leaf_nodes":   trial.suggest_int("max_leaf_nodes", 15, 127),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_bins":         trial.suggest_int("max_bins", 64, 255),
    }

    clf = HistGradientBoostingClassifier(
        **params,
        random_state=42,
        class_weight="balanced",
        early_stopping=False,
    )

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_f1s = []
    for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx],   y_train[val_idx]
        df_va = pd.DataFrame({"target": y_va})

        # Subsample for speed if large
        if len(X_tr) > 300_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tr), 300_000, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        y_pred = (probs >= 0.5).astype(int)
        f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)

    return float(np.mean(fold_f1s))


def run_bayesian_optimization():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 4: Bayesian Hyperparameter Optimization (Optuna)")
    print("=" * 60)

    # ── Load features ────────────────────────────────────────────
    print("\nLoading feature tables...")
    train_df = pd.read_csv(TRAIN_FEATURES)
    test_df  = pd.read_csv(TEST_FEATURES)

    X_train      = train_df[FEATURE_COLUMNS].values
    y_train      = train_df["target"].values
    groups_train = train_df["trip_id"].values

    X_test  = test_df[FEATURE_COLUMNS].values
    y_test  = test_df["target"].values

    print(f"Train: {X_train.shape[0]:,} rows from {train_df['trip_id'].nunique()} trips")
    print(f"Test:  {X_test.shape[0]:,} rows from {test_df['trip_id'].nunique()} trips\n")

    # ── Baseline (untuned) evaluation ────────────────────────────
    print("Evaluating UNTUNED HistGradientBoosting baseline...")
    untuned_path = os.path.join(MODEL_DIR, "HistGradientBoosting.joblib")
    if os.path.exists(untuned_path):
        untuned_model = joblib.load(untuned_path)
        print(f"  Loaded pre-trained model from {untuned_path}")
    else:
        untuned_model = HistGradientBoostingClassifier(
            max_iter=80, max_depth=8, learning_rate=0.1,
            random_state=42, class_weight="balanced"
        )
        untuned_model.fit(X_train, y_train)

    untuned_probs = untuned_model.predict_proba(X_test)[:, 1]
    untuned_metrics = compute_observation_metrics(test_df, untuned_probs)
    print(f"  Untuned – Top-1 Acc: {untuned_metrics['obs_top1_accuracy']:.4f}  "
          f"Macro F1: {untuned_metrics['cand_macro_f1']:.4f}  "
          f"LogLoss: {untuned_metrics['cand_log_loss']:.4f}")

    # ── Optuna study ──────────────────────────────────────────────
    db_path = os.path.join(OUT_DIR, "optuna_study.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="pathsnap_hgb_tuning",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    n_existing = len(study.trials)
    n_needed   = max(0, N_TRIALS - n_existing)
    print(f"\nRunning {n_needed} new Optuna trials ({n_existing} already in DB)...")
    t_opt_start = time.time()

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups_train),
        n_trials=n_needed,
        show_progress_bar=True,
        n_jobs=1,
    )
    t_opt_elapsed = time.time() - t_opt_start
    print(f"Optimization complete in {t_opt_elapsed:.1f}s")

    # ── Best trial summary ────────────────────────────────────────
    best = study.best_trial
    print(f"\nBest Trial #{best.number}: CV Macro F1 = {best.value:.4f}")
    print("Best Hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save best params as JSON
    best_params_path = os.path.join(OUT_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({"best_cv_macro_f1": best.value, "params": best.params}, f, indent=2)
    print(f"Saved best params to {best_params_path}")

    # ── Save trials history CSV ───────────────────────────────────
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(OUT_DIR, "trials_history.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved {len(trials_df)} trial records to {trials_csv}")

    # ── Refit tuned model on ALL training data ────────────────────
    print("\nRefitting TUNED model on all training data...")
    tuned_model = HistGradientBoostingClassifier(
        **best.params,
        random_state=42,
        class_weight="balanced",
        early_stopping=False,
    )
    t_fit = time.time()
    tuned_model.fit(X_train, y_train)
    t_fit_elapsed = time.time() - t_fit
    print(f"  Fit time: {t_fit_elapsed:.1f}s")

    tuned_model_path = os.path.join(MODEL_DIR, "HistGradientBoosting_tuned.joblib")
    joblib.dump(tuned_model, tuned_model_path)
    print(f"  Saved tuned model to {tuned_model_path}")

    # ── Tuned evaluation on student-test ─────────────────────────
    tuned_probs   = tuned_model.predict_proba(X_test)[:, 1]
    tuned_metrics = compute_observation_metrics(test_df, tuned_probs)
    print(f"\nTuned – Top-1 Acc: {tuned_metrics['obs_top1_accuracy']:.4f}  "
          f"Macro F1: {tuned_metrics['cand_macro_f1']:.4f}  "
          f"LogLoss: {tuned_metrics['cand_log_loss']:.4f}")

    # ── Comparison table ──────────────────────────────────────────
    comparison = pd.DataFrame([
        {
            "Configuration": "Untuned (default)",
            "Student-Test Top-1 Accuracy": untuned_metrics["obs_top1_accuracy"],
            "Student-Test Macro F1":       untuned_metrics["cand_macro_f1"],
            "Student-Test Log Loss":        untuned_metrics["cand_log_loss"],
            "Best CV Macro F1":             "—",
        },
        {
            "Configuration": f"Tuned (Optuna {N_TRIALS} trials)",
            "Student-Test Top-1 Accuracy": tuned_metrics["obs_top1_accuracy"],
            "Student-Test Macro F1":       tuned_metrics["cand_macro_f1"],
            "Student-Test Log Loss":        tuned_metrics["cand_log_loss"],
            "Best CV Macro F1":             round(best.value, 4),
        },
    ])
    comp_path = os.path.join(OUT_DIR, "tuned_vs_untuned.csv")
    comparison.to_csv(comp_path, index=False)
    print(f"\nSaved Tuned vs Untuned comparison to {comp_path}")

    try:
        print("\n" + comparison.to_markdown(index=False))
    except Exception:
        print("\n" + comparison.to_string(index=False))

    # ── Generate Optuna plots ─────────────────────────────────────
    _save_optuna_plots(study, OUT_DIR)

    return study, tuned_model, tuned_probs


def _save_optuna_plots(study, out_dir: str):
    """Save Optuna diagnostic plots as PNG files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Optimization history
        try:
            from optuna.visualization.matplotlib import plot_optimization_history
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_optimization_history(study, ax=ax)
            ax.set_title("Optuna Optimization History – HistGradientBoosting")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "optimization_history.png"), dpi=120)
            plt.close(fig)
            print("Saved optimization_history.png")
        except Exception as e:
            print(f"  optimization_history plot failed: {e}")

        # Parameter importances
        try:
            from optuna.visualization.matplotlib import plot_param_importances
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_param_importances(study, ax=ax)
            ax.set_title("Hyperparameter Importances")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "param_importances.png"), dpi=120)
            plt.close(fig)
            print("Saved param_importances.png")
        except Exception as e:
            print(f"  param_importances plot failed: {e}")

        # Contour plots for top-2 params
        try:
            from optuna.visualization.matplotlib import plot_contour
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_contour(study, params=["learning_rate", "max_leaf_nodes"], ax=ax)
            ax.set_title("Contour: learning_rate vs max_leaf_nodes")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "contour_lr_leaves.png"), dpi=120)
            plt.close(fig)
            print("Saved contour_lr_leaves.png")
        except Exception as e:
            print(f"  contour plot failed: {e}")

        # Manual history plot as fallback
        trials_df = study.trials_dataframe()
        if "value" in trials_df.columns:
            best_so_far = trials_df["value"].cummax()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trials_df["number"], trials_df["value"], "o", alpha=0.4, markersize=4, label="Trial F1")
            ax.plot(trials_df["number"], best_so_far, "-", color="red", linewidth=2, label="Best so far")
            ax.set_xlabel("Trial")
            ax.set_ylabel("CV Macro F1")
            ax.set_title("Bayesian Optimization History")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "optimization_history_manual.png"), dpi=120)
            plt.close(fig)
            print("Saved optimization_history_manual.png")

    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    run_bayesian_optimization()
