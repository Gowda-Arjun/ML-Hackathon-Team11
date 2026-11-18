"""Dataset exploration, inventory generation, and statistics collection."""

import os
import sys
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.loader import load_trip, get_available_trip_ids


def compute_inventory(base_data_dir: str = "Student folder") -> pd.DataFrame:
    """Compute complete inventory across train and student_test partitions."""
    partitions = ["train", "student_test"]
    records = []

    for part in partitions:
        part_dir = os.path.join(base_data_dir, part)
        trip_ids = get_available_trip_ids(part_dir)

        for trip_id in trip_ids:
            trip_path = os.path.join(part_dir, trip_id)
            trip = load_trip(trip_path, trip_id, load_route=True)

            # Observations & sampling intervals
            n_obs = len(trip.track)
            n_nodes = len(trip.nodes)
            n_arcs = len(trip.arcs)
            n_route = len(trip.route) if trip.route is not None else 0

            # Missing or null values
            null_track = trip.track.isnull().sum().sum()
            null_nodes = trip.nodes.isnull().sum().sum()
            null_arcs = trip.arcs.isnull().sum().sum()

            # Sampling intervals
            dt = np.diff(trip.track["t"].values) if n_obs > 1 else np.array([0])
            median_dt = float(np.median(dt))
            max_dt = float(np.max(dt))
            min_dt = float(np.min(dt))
            pct_1s = float((dt == 1.0).mean() * 100) if len(dt) > 0 else 0.0

            # Coordinate bounds
            min_lon, max_lon = float(trip.track["lon"].min()), float(trip.track["lon"].max())
            min_lat, max_lat = float(trip.track["lat"].min()), float(trip.track["lat"].max())

            # Route connectivity check
            route_connected_pct = 0.0
            if n_route > 1:
                u_seq = trip.arcs.iloc[trip.route]["u"].values
                v_seq = trip.arcs.iloc[trip.route]["v"].values
                connected_count = np.sum(v_seq[:-1] == u_seq[1:])
                route_connected_pct = float(connected_count / (n_route - 1) * 100)

            records.append({
                "partition": part,
                "trip_id": trip_id,
                "track_points": n_obs,
                "road_nodes": n_nodes,
                "road_arcs": n_arcs,
                "route_arcs": n_route,
                "null_values": int(null_track + null_nodes + null_arcs),
                "dt_median_s": median_dt,
                "dt_max_s": max_dt,
                "pct_dt_1s": round(pct_1s, 2),
                "route_continuity_pct": round(route_connected_pct, 2),
                "lon_range": f"[{min_lon:.4f}, {max_lon:.4f}]",
                "lat_range": f"[{min_lat:.4f}, {max_lat:.4f}]",
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_csv = "artifacts/results/dataset_inventory.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    print("Computing dataset inventory across all trips...")
    inv_df = compute_inventory()
    inv_df.to_csv(out_csv, index=False)
    print(f"Saved dataset inventory to {out_csv}")
    print("\nSummary Statistics by Partition:")
    summary = inv_df.groupby("partition").agg(
        num_trips=("trip_id", "count"),
        total_track_points=("track_points", "sum"),
        mean_track_points=("track_points", "mean"),
        total_road_arcs=("road_arcs", "sum"),
        mean_dt_1s_pct=("pct_dt_1s", "mean"),
        mean_route_continuity=("route_continuity_pct", "mean"),
    )
    print(summary)
