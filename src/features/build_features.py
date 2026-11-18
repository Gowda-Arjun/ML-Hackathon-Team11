"""Feature extraction pipeline for PathSnap candidate classification."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from src.data.loader import TripData, load_trip, get_available_trip_ids
from src.features.spatial import LocalMetricProjector, compute_angle_difference
from src.features.alignment import align_route_to_observations
from src.features.candidates import RoadNetworkIndex


FEATURE_COLUMNS = [
    "perp_dist",         # Perpendicular distance from GPS to segment (m)
    "heading_diff",      # Absolute angle difference between GPS heading and arc (rad)
    "along_ratio",       # Projection ratio along segment [0, 1]
    "dist_to_start",     # Distance from GPS to from_node u (m)
    "dist_to_end",       # Distance from GPS to to_node v (m)
    "seg_length",        # Length of candidate segment (m)
    "speed_gps",         # Instantaneous GPS speed (m/s)
    "dt",                # Sampling time difference (s)
    "dist_rank",         # Rank of candidate sorted by perpendicular distance (0..k-1)
]


def extract_features_for_trip(
    trip: TripData,
    k_candidates: int = 8,
    k_pool: int = 80,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Extract candidate feature table and summary metadata for a single trip."""
    center_lon = float(trip.track["lon"].mean())
    center_lat = float(trip.track["lat"].mean())
    projector = LocalMetricProjector(center_lon, center_lat)

    # Convert GPS track to local metric coordinates
    pts = projector.forward(trip.track["lon"].values, trip.track["lat"].values)
    n_pts = len(pts)
    timestamps = trip.track["t"].values

    # GPS headings and speeds
    dx = np.diff(pts[:, 0], prepend=pts[0, 0])
    dy = np.diff(pts[:, 1], prepend=pts[0, 1])
    dt = np.diff(timestamps, prepend=timestamps[0] - 1.0)
    dt = np.where(dt <= 0.0, 1.0, dt)  # guard against non-positive dt

    # Handle first point: use movement towards second point
    if n_pts > 1:
        dx[0] = pts[1, 0] - pts[0, 0]
        dy[0] = pts[1, 1] - pts[0, 1]
        dt[0] = max(timestamps[1] - timestamps[0], 1.0)

    speeds = np.sqrt(dx * dx + dy * dy) / dt
    gps_headings = np.arctan2(dy, dx)

    # Reference route alignment if available
    ref_arcs, _ = align_route_to_observations(trip, projector) if (trip.route is not None) else (np.full(n_pts, -1), None)

    # Build road network spatial index
    road_index = RoadNetworkIndex(trip, projector)

    records = []
    miss_count = 0
    valid_ref_count = 0

    for t in range(n_pts):
        pt = pts[t:t+1]
        cand_arcs, geom = road_index.get_candidates(pt, k_pool=k_pool, k_cand=k_candidates)

        true_arc = ref_arcs[t]
        has_true = true_arc in cand_arcs
        if true_arc != -1:
            valid_ref_count += 1
            if not has_true:
                miss_count += 1

        cand_headings = geom["seg_heading"]
        heading_diffs = compute_angle_difference(gps_headings[t], cand_headings)

        for rank, arc_id in enumerate(cand_arcs):
            is_match = 1 if (arc_id == true_arc) else 0
            records.append({
                "trip_id": trip.trip_id,
                "obs_idx": t,
                "arc_id": arc_id,
                "target": is_match,
                "perp_dist": geom["perp_dist"][rank],
                "heading_diff": heading_diffs[rank],
                "along_ratio": geom["along_ratio"][rank],
                "dist_to_start": geom["dist_to_start"][rank],
                "dist_to_end": geom["dist_to_end"][rank],
                "seg_length": geom["seg_length"][rank],
                "speed_gps": speeds[t],
                "dt": dt[t],
                "dist_rank": rank,
            })

    df = pd.DataFrame(records)
    meta = {
        "trip_id": trip.trip_id,
        "n_obs": n_pts,
        "valid_ref_count": valid_ref_count,
        "miss_count": miss_count,
        "miss_rate": (miss_count / valid_ref_count) if valid_ref_count > 0 else 0.0,
    }
    return df, meta


def build_dataset_features(
    partition: str = "train",
    base_dir: str = "Student folder",
    max_trips: int = None,
    k_candidates: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate features across all trips in a partition."""
    part_dir = os.path.join(base_dir, partition)
    trip_ids = get_available_trip_ids(part_dir)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    dfs = []
    metas = []
    print(f"Extracting features for {len(trip_ids)} trips in '{partition}'...")

    for i, trip_id in enumerate(trip_ids):
        trip_path = os.path.join(part_dir, trip_id)
        trip = load_trip(trip_path, trip_id, load_route=True)
        feat_df, meta = extract_features_for_trip(trip, k_candidates=k_candidates)
        dfs.append(feat_df)
        metas.append(meta)
        if (i + 1) % 10 == 0 or (i + 1) == len(trip_ids):
            print(f"  Processed {i+1}/{len(trip_ids)} trips...")

    full_df = pd.concat(dfs, ignore_index=True)
    meta_df = pd.DataFrame(metas)
    return full_df, meta_df


if __name__ == "__main__":
    out_dir = "artifacts/results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Build Student Test features (10 trips)
    print("--- Building Student Test Features ---")
    test_features, test_meta = build_dataset_features("student_test")
    test_features.to_csv(f"{out_dir}/student_test_features.csv.gz", compression="gzip", index=False)
    test_meta.to_csv(f"{out_dir}/student_test_candidate_summary.csv", index=False)
    print(f"Student Test shape: {test_features.shape}")
    print(f"Student Test Mean Candidate Miss Rate: {test_meta['miss_rate'].mean() * 100:.2f}%\n")

    # 2. Build Training features (80 trips)
    print("--- Building Training Features ---")
    train_features, train_meta = build_dataset_features("train")
    train_features.to_csv(f"{out_dir}/train_features.csv.gz", compression="gzip", index=False)
    train_meta.to_csv(f"{out_dir}/train_candidate_summary.csv", index=False)
    print(f"Train shape: {train_features.shape}")
    print(f"Train Mean Candidate Miss Rate: {train_meta['miss_rate'].mean() * 100:.2f}%")
