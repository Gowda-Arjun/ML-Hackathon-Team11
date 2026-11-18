"""Data loading utilities for PathSnap dataset."""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


@dataclass
class TripData:
    trip_id: str
    track: pd.DataFrame       # columns: ['lon', 'lat', 't']
    nodes: pd.DataFrame       # columns: ['lon', 'lat'], index: node_id (0..N-1)
    arcs: pd.DataFrame        # columns: ['u', 'v'], index: arc_id (0..M-1)
    route: Optional[np.ndarray] = None  # array of ground-truth arc_ids traversed


def load_trip(trip_dir: str, trip_id: str, load_route: bool = True) -> TripData:
    """Load all files for a single trip."""
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    # Load track (lon, lat, timestamp_sec)
    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
    )

    # Load nodes (lon, lat)
    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
    )

    # Load arcs (from_node_id u, to_node_id v)
    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["u", "v"],
        dtype={"u": np.int64, "v": np.int64},
    )

    # Load route if available
    route = None
    if load_route and os.path.exists(route_path):
        route_df = pd.read_csv(
            route_path,
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype={"arc_id": np.int64},
        )
        route = route_df["arc_id"].values

    return TripData(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def get_available_trip_ids(partition_dir: str) -> List[str]:
    """Return sorted list of trip IDs available in a partition folder."""
    if not os.path.exists(partition_dir):
        return []
    trips = [
        d for d in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, d)) and not d.startswith(".")
    ]
    return sorted(trips)
