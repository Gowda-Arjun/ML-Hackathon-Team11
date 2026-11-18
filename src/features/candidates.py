"""Road network candidate generation module using KDTree spatial indexing."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List, Dict
from src.data.loader import TripData
from src.features.spatial import LocalMetricProjector, point_to_segment_distances


class RoadNetworkIndex:
    """Spatial index over all road segments (arcs) in a trip network."""

    def __init__(self, trip: TripData, projector: LocalMetricProjector):
        self.trip = trip
        self.projector = projector

        # Convert node coordinates to local metric space
        self.node_pts = projector.forward(trip.nodes["lon"].values, trip.nodes["lat"].values)

        # Arc endpoints
        self.u_idx = trip.arcs["u"].values
        self.v_idx = trip.arcs["v"].values
        self.seg_starts = self.node_pts[self.u_idx]
        self.seg_ends = self.node_pts[self.v_idx]

        # Multi-point representation for each arc: start, midpoint, end
        midpoints = 0.5 * (self.seg_starts + self.seg_ends)
        index_points = np.vstack([self.seg_starts, midpoints, self.seg_ends])
        self.arc_ids_map = np.tile(np.arange(len(trip.arcs)), 3)

        # Build KDTree
        self.tree = cKDTree(index_points)

        # Build forward adjacency list for fast graph connectivity checks
        self.outgoing_arcs: Dict[int, List[int]] = {}
        for arc_id, (u, v) in enumerate(zip(self.u_idx, self.v_idx)):
            # Index by from_node u
            pass

    def get_candidates(
        self,
        pt: np.ndarray,            # (1, 2) [x, y]
        k_pool: int = 80,
        k_cand: int = 8,
    ) -> Tuple[np.ndarray, dict]:
        """Find top-k candidate road segments for an observation point.
        
        Returns:
          cand_arcs: array of arc_ids of shape (k_cand,)
          geom_dict: dict of geometric features for each candidate arc
        """
        _, query_idx = self.tree.query(pt, k=min(k_pool, len(self.arc_ids_map)))
        if query_idx.ndim == 1:
            q_ids = query_idx
        else:
            q_ids = query_idx[0]

        # Unique arc IDs from query
        cand_arcs_pool = np.unique(self.arc_ids_map[q_ids])

        starts = self.seg_starts[cand_arcs_pool]
        ends = self.seg_ends[cand_arcs_pool]
        rep_pt = np.repeat(pt, len(starts), axis=0)

        geom = point_to_segment_distances(rep_pt, starts, ends)
        sort_idx = np.argsort(geom["perp_dist"])[:k_cand]

        top_arcs = cand_arcs_pool[sort_idx]
        top_geom = {k: v[sort_idx] for k, v in geom.items()}

        return top_arcs, top_geom
