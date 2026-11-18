"""HMM Transition Probability Module.

Builds a NetworkX directed graph from road arcs and computes
transition probabilities between candidate arcs using network
shortest-path distances compared to GPS displacement.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

try:
    import networkx as nx
except ImportError:
    raise ImportError("Run: pip install networkx")

from src.data.loader import TripData
from src.features.spatial import LocalMetricProjector


class RoadGraph:
    """NetworkX directed graph for a road network. Computes arc-to-arc
    transition probabilities using graph shortest paths.
    """

    def __init__(self, trip: TripData, projector: LocalMetricProjector, beta: float = 20.0):
        """
        Args:
            trip:       TripData with nodes and arcs.
            projector:  LocalMetricProjector for this trip.
            beta:       Scale parameter for transition probability (meters).
                        Larger beta → smoother, more tolerant of detours.
        """
        self.beta = beta
        self.trip = trip
        self.projector = projector

        # Project nodes to metric coordinates
        node_pts = projector.forward(trip.nodes["lon"].values, trip.nodes["lat"].values)
        self.node_pts = node_pts  # (N_nodes, 2)

        # Build directed graph: nodes are road intersection IDs, edges are arcs
        self.G = nx.DiGraph()
        arcs = trip.arcs

        for arc_id, (u, v) in enumerate(zip(arcs["u"].values, arcs["v"].values)):
            # Edge weight = Euclidean arc length in meters
            length = float(np.linalg.norm(node_pts[v] - node_pts[u]))
            self.G.add_edge(u, v, weight=length, arc_id=arc_id)

        # Precompute arc endpoints (from_node, to_node) for quick lookup
        self.arc_u = arcs["u"].values   # (M,)
        self.arc_v = arcs["v"].values   # (M,)

        # Midpoints of each arc in metric space
        self.arc_midpoints = 0.5 * (node_pts[self.arc_u] + node_pts[self.arc_v])
        self._dist_cache: Dict[Tuple[int, int], float] = {}

    def direct_successors(self, arc_id: int) -> List[int]:
        """Return arc IDs that can be directly reached from arc_id (v[arc] == u[next])."""
        v_node = int(self.arc_v[arc_id])
        successors = []
        for _, w, data in self.G.out_edges(v_node, data=True):
            successors.append(data["arc_id"])
        return successors

    def _network_distance(self, from_arc: int, to_arc: int) -> float:
        """Shortest-path network distance between end of from_arc and start of to_arc (metres)."""
        u_start = int(self.arc_v[from_arc])  # end node of from_arc
        v_end   = int(self.arc_u[to_arc])    # start node of to_arc
        if u_start == v_end:
            return 0.0
        key = (u_start, v_end)
        if key in self._dist_cache:
            return self._dist_cache[key]
        try:
            dist = float(nx.shortest_path_length(self.G, u_start, v_end, weight="weight"))
            self._dist_cache[key] = dist
            return dist
        except nx.NetworkXNoPath:
            self._dist_cache[key] = float("inf")
            return float("inf")

    def transition_log_prob(
        self,
        from_arc: int,
        to_arc: int,
        gps_displacement: float,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute log P(to_arc | from_arc) using graph-distance formulation:
            P(to_arc | from_arc) ∝ exp(-|d_net - d_gps| / beta)
        """
        # Staying on the same road arc
        if from_arc == to_arc:
            return 0.0

        # Direct road graph connection: v[from_arc] == u[to_arc]
        if int(self.arc_v[from_arc]) == int(self.arc_u[to_arc]):
            return 0.0

        d_net = self._network_distance(from_arc, to_arc)
        if np.isinf(d_net):
            return np.log(eps)  # disconnected penalty

        diff = abs(d_net - gps_displacement)
        return -diff / self.beta  # unnormalized log-prob

    def compute_transition_matrix(
        self,
        cand_arcs_t: np.ndarray,     # candidate arcs at step t
        cand_arcs_next: np.ndarray,  # candidate arcs at step t+1
        gps_displacement: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute transition log-probability matrix between candidate sets.

        Returns:
          log_trans: (K, K) matrix where log_trans[i, j] = log P(cand_arcs_next[j] | cand_arcs_t[i])
        """
        K1 = len(cand_arcs_t)
        K2 = len(cand_arcs_next)
        log_trans = np.full((K1, K2), np.log(eps), dtype=np.float64)

        for i, ai in enumerate(cand_arcs_t):
            direct = set(self.direct_successors(ai)) | {ai}
            for j, aj in enumerate(cand_arcs_next):
                if aj in direct:
                    # Directly connected – high probability
                    log_trans[i, j] = self.transition_log_prob(
                        ai, aj, gps_displacement, eps
                    )
                else:
                    # Need full shortest-path check
                    log_trans[i, j] = self.transition_log_prob(
                        ai, aj, gps_displacement, eps
                    )

        return log_trans
