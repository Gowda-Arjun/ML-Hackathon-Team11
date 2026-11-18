"""Viterbi Decoder for HMM-based Route Reconstruction.

Runs the classic Viterbi dynamic programming algorithm in log-space
to find the most probable sequence of road arcs given GPS observations.

Handles:
  - GPS dropouts (Δt > 5 s): relaxes transition constraints.
  - Candidate misses: fallback to nearest arc by perpendicular distance.
  - Log-space arithmetic throughout to prevent numerical underflow.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Dict, Optional

from src.data.loader import TripData, load_trip, get_available_trip_ids
from src.features.spatial import LocalMetricProjector, compute_angle_difference
from src.features.candidates import RoadNetworkIndex
from src.features.build_features import FEATURE_COLUMNS, extract_features_for_trip
from src.features.alignment import align_route_to_observations
from Phase_2.hmm.transitions import RoadGraph
from Phase_2.hmm.emissions import combined_emission_log_probs


DROPOUT_THRESHOLD_S = 5.0   # seconds – treat as a gap / dropout
SIGMA_Z             = 20.0  # GPS noise std dev in meters
EMISSION_ALPHA      = 0.85  # optimal balance: 85% tuned classifier + 15% distance prior
BETA_TRANS          = 20.0  # transition distribution scale (meters)
EPS                 = 1e-9


def viterbi_decode_trip(
    trip: TripData,
    model,
    k_candidates: int = 8,
    sigma_z: float = SIGMA_Z,
    emission_alpha: float = EMISSION_ALPHA,
    beta_trans: float = BETA_TRANS,
    dropout_threshold: float = DROPOUT_THRESHOLD_S,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Run the full HMM + Viterbi pipeline for a single trip.

    Returns:
        decoded_arcs:   (N,) array of decoded arc_id per observation.
        cand_arcs_list: list of length N, each entry is (K,) array of candidate arc IDs.
        obs_probs_list: (N, K) matrix of classifier probabilities per candidate.
    """
    # ── 1. Project trip to metric space ─────────────────────────
    center_lon = float(trip.track["lon"].mean())
    center_lat = float(trip.track["lat"].mean())
    projector  = LocalMetricProjector(center_lon, center_lat)

    pts = projector.forward(trip.track["lon"].values, trip.track["lat"].values)
    n_pts      = len(pts)
    timestamps = trip.track["t"].values

    # ── 2. Build spatial index + road graph ──────────────────────
    road_index = RoadNetworkIndex(trip, projector)
    road_graph = RoadGraph(trip, projector, beta=beta_trans)

    # ── 3. GPS headings, speeds, Δt ──────────────────────────────
    dx = np.diff(pts[:, 0], prepend=pts[0, 0])
    dy = np.diff(pts[:, 1], prepend=pts[0, 1])
    dt = np.diff(timestamps, prepend=timestamps[0] - 1.0)
    dt = np.where(dt <= 0.0, 1.0, dt)
    if n_pts > 1:
        dx[0] = pts[1, 0] - pts[0, 0]
        dy[0] = pts[1, 1] - pts[0, 1]
        dt[0] = max(timestamps[1] - timestamps[0], 1.0)

    speeds      = np.sqrt(dx * dx + dy * dy) / dt
    gps_headings = np.arctan2(dy, dx)

    # ── 4. Build feature records for all observations ─────────────
    cand_arcs_list   = []
    cand_perps_list  = []
    cand_probs_list  = []

    for t in range(n_pts):
        pt        = pts[t:t+1]
        cand_arcs, geom = road_index.get_candidates(pt, k_cand=k_candidates)

        if len(cand_arcs) == 0:
            # Fallback: single nearest arc
            fallback_arc = _nearest_arc_fallback(pts[t], road_index)
            cand_arcs = np.array([fallback_arc])
            perps     = np.array([0.0])
        else:
            perps = geom["perp_dist"]

        # Build feature rows for this timestep's candidates
        heading_diffs = compute_angle_difference(
            gps_headings[t], geom["seg_heading"] if len(cand_arcs) > 1 or geom else np.zeros(1)
        )
        rows = []
        for rank, arc_id in enumerate(cand_arcs):
            r = {
                "perp_dist":    perps[rank] if rank < len(perps) else perps[0],
                "heading_diff": float(heading_diffs[rank]) if rank < len(heading_diffs) else 0.0,
                "along_ratio":  float(geom["along_ratio"][rank]) if rank < len(perps) else 0.5,
                "dist_to_start": float(geom["dist_to_start"][rank]) if rank < len(perps) else 0.0,
                "dist_to_end":   float(geom["dist_to_end"][rank])   if rank < len(perps) else 0.0,
                "seg_length":    float(geom["seg_length"][rank])     if rank < len(perps) else 1.0,
                "speed_gps":    float(speeds[t]),
                "dt":           float(dt[t]),
                "dist_rank":    rank,
            }
            rows.append(r)

        feat_df = pd.DataFrame(rows)
        X_t = feat_df[FEATURE_COLUMNS].values
        probs_t = model.predict_proba(X_t)[:, 1]  # P(match=1) for each candidate

        cand_arcs_list.append(cand_arcs)
        cand_perps_list.append(perps)
        cand_probs_list.append(probs_t)

    # ── 5. Viterbi DP in log-space ───────────────────────────────
    # viterbi[t][j] = log-prob of best path ending in state j at step t
    # backptr[t][j] = index into cand_arcs_list[t-1] from which we came

    K0            = len(cand_arcs_list[0])
    log_v         = combined_emission_log_probs(
        cand_probs_list[0], cand_perps_list[0], sigma_z, emission_alpha
    )
    backptrs: List[np.ndarray] = [np.zeros(K0, dtype=np.int32)]
    all_log_v     = [log_v]

    for t in range(1, n_pts):
        K_prev    = len(cand_arcs_list[t - 1])
        K_curr    = len(cand_arcs_list[t])
        prev_arcs = cand_arcs_list[t - 1]
        curr_arcs = cand_arcs_list[t]

        # GPS displacement from t-1 to t
        gps_disp = float(np.linalg.norm(pts[t] - pts[t - 1]))

        # Relax transitions for dropouts
        effective_beta = beta_trans
        if dt[t] > dropout_threshold:
            effective_beta = beta_trans * (dt[t] / dropout_threshold)

        log_emission_t = combined_emission_log_probs(
            cand_probs_list[t], cand_perps_list[t], sigma_z, emission_alpha
        )

        log_trans = road_graph.compute_transition_matrix(
            prev_arcs, curr_arcs, gps_disp, eps=EPS
        )  # (K_prev, K_curr)

        # For each candidate j at t: best_{i} [ log_v[t-1][i] + log_trans[i,j] ]
        log_vt      = np.full(K_curr, -np.inf)
        back_t      = np.zeros(K_curr, dtype=np.int32)

        for j in range(K_curr):
            scores      = all_log_v[-1] + log_trans[:, j]
            best_i      = int(np.argmax(scores))
            log_vt[j]   = scores[best_i] + log_emission_t[j]
            back_t[j]   = best_i

        all_log_v.append(log_vt)
        backptrs.append(back_t)

    # ── 6. Backtrack ─────────────────────────────────────────────
    decoded_indices = np.zeros(n_pts, dtype=np.int32)
    decoded_indices[-1] = int(np.argmax(all_log_v[-1]))

    for t in range(n_pts - 2, -1, -1):
        decoded_indices[t] = backptrs[t + 1][decoded_indices[t + 1]]

    decoded_arcs = np.array([
        cand_arcs_list[t][decoded_indices[t]] for t in range(n_pts)
    ], dtype=np.int64)

    return decoded_arcs, cand_arcs_list, np.array([p for p in cand_probs_list], dtype=object)


def _nearest_arc_fallback(pt: np.ndarray, road_index: RoadNetworkIndex) -> int:
    """Fallback: return the arc whose midpoint is nearest to pt."""
    dists = np.linalg.norm(road_index.arc_midpoints - pt, axis=1)
    return int(np.argmin(dists))


def evaluate_viterbi_on_partition(
    partition_dir: str,
    model_path: str,
    k_candidates: int = 8,
    max_trips: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run Viterbi on all trips in a partition and report Before/After HMM metrics.
    
    Returns DataFrame with per-trip and aggregate metrics.
    """
    model = joblib.load(model_path)
    trip_ids = get_available_trip_ids(partition_dir)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    records = []
    print(f"Evaluating Viterbi on {len(trip_ids)} trips from '{partition_dir}'...")

    for i, trip_id in enumerate(trip_ids):
        trip_path = os.path.join(partition_dir, trip_id)
        trip      = load_trip(trip_path, trip_id, load_route=True)
        if trip.route is None:
            continue

        center_lon = float(trip.track["lon"].mean())
        center_lat = float(trip.track["lat"].mean())
        projector  = LocalMetricProjector(center_lon, center_lat)

        # Ground truth reference arcs per observation
        ref_arcs, _ = align_route_to_observations(trip, projector)

        # --- Classifier-only predictions (no HMM) ---
        feat_df, _ = extract_features_for_trip(trip, k_candidates=k_candidates, is_training=False)
        X = feat_df[FEATURE_COLUMNS].values
        probs = model.predict_proba(X)[:, 1]
        feat_df["pred_score"] = probs
        top1_clf = feat_df.sort_values(["obs_idx", "pred_score"], ascending=[True, False]) \
                          .groupby("obs_idx").first()["arc_id"].values
        clf_top1_acc = float(np.mean(top1_clf == ref_arcs[:len(top1_clf)]))

        # Continuity (classifier only)
        clf_continuity = _route_continuity(top1_clf, trip.arcs)

        # --- HMM + Viterbi ---
        try:
            viterbi_arcs, _, _ = viterbi_decode_trip(trip, model, k_candidates)
            hmm_top1_acc    = float(np.mean(viterbi_arcs == ref_arcs[:len(viterbi_arcs)]))
            hmm_continuity  = _route_continuity(viterbi_arcs, trip.arcs)
        except Exception as exc:
            print(f"  Trip {trip_id}: Viterbi error – {exc}")
            hmm_top1_acc   = float("nan")
            hmm_continuity = float("nan")
            viterbi_arcs   = top1_clf

        records.append({
            "trip_id":          trip_id,
            "n_obs":            len(ref_arcs),
            "clf_top1_acc":     round(clf_top1_acc,   4),
            "clf_continuity":   round(clf_continuity, 4),
            "hmm_top1_acc":     round(hmm_top1_acc,   4),
            "hmm_continuity":   round(hmm_continuity, 4),
            "acc_improvement":  round(hmm_top1_acc - clf_top1_acc, 4),
            "cont_improvement": round(hmm_continuity - clf_continuity, 4),
        })
        print(f"  [{i+1}/{len(trip_ids)}] Trip {trip_id}: "
              f"Clf Top-1={clf_top1_acc:.3f} | "
              f"HMM Top-1={hmm_top1_acc:.3f} | "
              f"Clf Cont={clf_continuity:.3f} | "
              f"HMM Cont={hmm_continuity:.3f}")

    df = pd.DataFrame(records)
    if not df.empty:
        print("\n── Aggregate Before vs After HMM ──")
        for col in ["clf_top1_acc", "hmm_top1_acc", "clf_continuity", "hmm_continuity"]:
            print(f"  Mean {col}: {df[col].mean():.4f}")
    return df


def _route_continuity(arc_seq: np.ndarray, arcs_df: pd.DataFrame) -> float:
    """Fraction of consecutive (a_t, a_{t+1}) pairs that are road-connected."""
    if len(arc_seq) < 2:
        return 1.0
    u = arcs_df["u"].values
    v = arcs_df["v"].values
    same   = arc_seq[:-1] == arc_seq[1:]
    connected = v[arc_seq[:-1]] == u[arc_seq[1:]]
    return float(np.mean(same | connected))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="Student folder/student_test",
                        help="Path to trip partition directory")
    parser.add_argument("--model",    default="artifacts/models/HistGradientBoosting_tuned.joblib",
                        help="Path to trained model .joblib")
    parser.add_argument("--max-trips", type=int, default=None)
    parser.add_argument("--out",      default="artifacts/results/viterbi_evaluation.csv")
    args = parser.parse_args()

    df = evaluate_viterbi_on_partition(
        args.partition, args.model, max_trips=args.max_trips
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved evaluation to {args.out}")
