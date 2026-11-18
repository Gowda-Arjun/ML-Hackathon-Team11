"""Evaluation metrics for PathSnap map-matching and candidate classification."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score


def compute_observation_metrics(
    df: pd.DataFrame,
    pred_scores: np.ndarray,
    trip_arcs_df: pd.DataFrame = None,
) -> Dict[str, float]:
    """Calculate observation-level and candidate-level evaluation metrics.
    
    Args:
      df: DataFrame containing ['trip_id', 'obs_idx', 'arc_id', 'target']
      pred_scores: predicted probabilities P(y=1) for each candidate row in df
      trip_arcs_df: Optional DataFrame with ['u', 'v'] indexed by arc_id to check road continuity
    """
    df = df.copy()
    df["pred_score"] = pred_scores

    # Candidate-level binary metrics
    y_true = df["target"].values
    y_pred_binary = (pred_scores >= 0.5).astype(int)

    cand_acc = float(accuracy_score(y_true, y_pred_binary))
    cand_macro_f1 = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))
    cand_precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
    cand_recall = float(recall_score(y_true, y_pred_binary, zero_division=0))

    try:
        # Clip probabilities to avoid infinite log_loss
        clipped_scores = np.clip(pred_scores, 1e-6, 1 - 1e-6)
        cand_log_loss = float(log_loss(y_true, clipped_scores))
    except Exception:
        cand_log_loss = float("nan")

    # Observation-level Top-1 Accuracy: for each (trip_id, obs_idx), choose candidate with max score
    top_cands = df.sort_values(["trip_id", "obs_idx", "pred_score"], ascending=[True, True, False])
    top_cands = top_cands.groupby(["trip_id", "obs_idx"]).first().reset_index()

    # If the top candidate has target == 1, then Top-1 is correct
    obs_top1_acc = float(top_cands["target"].mean())

    # Route Continuity Rate across consecutive observations
    continuity_rates = []
    if trip_arcs_df is not None:
        for trip_id, trip_group in top_cands.groupby("trip_id"):
            trip_group = trip_group.sort_values("obs_idx")
            chosen_arcs = trip_group["arc_id"].values
            n_trans = len(chosen_arcs) - 1
            if n_trans <= 0:
                continue

            u_nodes = trip_arcs_df.iloc[chosen_arcs]["u"].values
            v_nodes = trip_arcs_df.iloc[chosen_arcs]["v"].values

            # Connected if same arc OR v[t-1] == u[t]
            same_arc = chosen_arcs[:-1] == chosen_arcs[1:]
            graph_connected = v_nodes[:-1] == u_nodes[1:]
            is_valid = same_arc | graph_connected
            continuity_rates.append(float(np.mean(is_valid)))

    mean_continuity = float(np.mean(continuity_rates)) if continuity_rates else float("nan")

    return {
        "obs_top1_accuracy": round(obs_top1_acc, 4),
        "cand_macro_f1": round(cand_macro_f1, 4),
        "cand_precision": round(cand_precision, 4),
        "cand_recall": round(cand_recall, 4),
        "cand_log_loss": round(cand_log_loss, 4),
        "route_continuity": round(mean_continuity, 4) if not np.isnan(mean_continuity) else 0.0,
    }
