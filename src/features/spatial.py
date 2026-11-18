"""Spatial geometry and local metric projection utilities."""

import numpy as np


class LocalMetricProjector:
    """Projects geographic coordinates (lon, lat in degrees) to local metric coordinates (x, y in meters).
    Uses a local tangent equirectangular projection centered at the trip center.
    """
    def __init__(self, center_lon: float, center_lat: float):
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.r_earth = 6371008.8  # Earth mean radius in meters
        self.lat_rad = np.radians(center_lat)
        self.cos_lat = np.cos(self.lat_rad)

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Convert lon, lat to local x, y in meters."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        x = np.radians(lon - self.center_lon) * self.r_earth * self.cos_lat
        y = np.radians(lat - self.center_lat) * self.r_earth
        return np.column_stack([x, y])

    def inverse(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convert local x, y in meters back to lon, lat."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lon = np.degrees(x / (self.r_earth * self.cos_lat)) + self.center_lon
        lat = np.degrees(y / self.r_earth) + self.center_lat
        return np.column_stack([lon, lat])


def point_to_segment_distances(
    points: np.ndarray,      # (N, 2) [x, y]
    seg_starts: np.ndarray,  # (N, 2) [x_a, y_a]
    seg_ends: np.ndarray     # (N, 2) [x_b, y_b]
) -> dict:
    """Vectorized calculation of point-to-segment geometric features.
    
    Returns dict of:
      - 'perp_dist': perpendicular distance from point to segment in meters
      - 'along_ratio': clamped projection ratio in [0, 1]
      - 'dist_to_start': Euclidean distance from point to start node (u)
      - 'dist_to_end': Euclidean distance from point to end node (v)
      - 'seg_length': length of segment in meters
      - 'seg_heading': heading angle of segment in radians [-pi, pi]
    """
    v = seg_ends - seg_starts  # (N, 2)
    w = points - seg_starts    # (N, 2)

    l2 = np.sum(v * v, axis=1)  # (N,)
    zero_mask = l2 == 0.0

    # Avoid div by zero
    safe_l2 = np.where(zero_mask, 1.0, l2)
    proj = np.sum(w * v, axis=1) / safe_l2
    along_ratio = np.clip(proj, 0.0, 1.0)
    along_ratio = np.where(zero_mask, 0.0, along_ratio)

    proj_pts = seg_starts + along_ratio[:, None] * v
    diff = points - proj_pts
    perp_dist = np.sqrt(np.sum(diff * diff, axis=1))

    diff_u = points - seg_starts
    dist_to_start = np.sqrt(np.sum(diff_u * diff_u, axis=1))

    diff_v = points - seg_ends
    dist_to_end = np.sqrt(np.sum(diff_v * diff_v, axis=1))

    seg_length = np.sqrt(l2)
    seg_heading = np.arctan2(v[:, 1], v[:, 0])

    return {
        "perp_dist": perp_dist,
        "along_ratio": along_ratio,
        "dist_to_start": dist_to_start,
        "dist_to_end": dist_to_end,
        "seg_length": seg_length,
        "seg_heading": seg_heading,
    }


def compute_angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """Calculate minimal absolute angle difference in radians in [0, pi]."""
    diff = np.abs(angle1 - angle2)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return diff
