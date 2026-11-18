"""Ground truth route-to-observation alignment module.
Associates each GPS observation with its reference road segment along the traversed route.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Tuple
from src.data.loader import TripData
from src.features.spatial import LocalMetricProjector, point_to_segment_distances


def align_route_to_observations(
    trip: TripData,
    projector: LocalMetricProjector,
    forward_window: int = 15,
    backward_window: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monotonically project GPS observations onto the ordered route arcs.
    
    Returns:
      ref_arcs: array of length N (number of track observations) with reference arc_id for each observation.
      ref_dists: array of length N with distance (meters) from observation to matched arc.
    """
    if trip.route is None or len(trip.route) == 0:
        return np.full(len(trip.track), -1, dtype=np.int64), np.full(len(trip.track), np.nan)

    # Project track points to local metric coordinates
    pts = projector.forward(trip.track["lon"].values, trip.track["lat"].values)
    n_pts = len(pts)

    # Project nodes to local metric coordinates
    node_pts = projector.forward(trip.nodes["lon"].values, trip.nodes["lat"].values)

    # Get endpoints for route arcs
    route_arcs = trip.route
    u_nodes = trip.arcs.iloc[route_arcs]["u"].values
    v_nodes = trip.arcs.iloc[route_arcs]["v"].values
    seg_starts = node_pts[u_nodes]
    seg_ends = node_pts[v_nodes]
    n_route = len(route_arcs)

    ref_arcs = np.zeros(n_pts, dtype=np.int64)
    ref_dists = np.zeros(n_pts, dtype=np.float64)

    curr_idx = 0
    for t in range(n_pts):
        pt = pts[t:t+1]  # (1, 2)
        win_start = max(0, curr_idx - backward_window)
        win_end = min(n_route, curr_idx + forward_window + 1)

        w_starts = seg_starts[win_start:win_end]
        w_ends = seg_ends[win_start:win_end]
        w_pts = np.repeat(pt, len(w_starts), axis=0)

        geom = point_to_segment_distances(w_pts, w_starts, w_ends)
        dists = geom["perp_dist"]

        best_local = np.argmin(dists)
        best_global = win_start + best_local
        curr_idx = best_global

        ref_arcs[t] = route_arcs[best_global]
        ref_dists[t] = dists[best_local]

    return ref_arcs, ref_dists
