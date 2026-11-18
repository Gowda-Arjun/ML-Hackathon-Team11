"""Folium Interactive Map Generator.

Produces 4-layer interactive HTML maps for each trip:
  Layer 1: Raw GPS observations (blue circle markers)
  Layer 2: Road network arcs (grey lines)
  Layer 3: Ground-truth route (thick green polyline)
  Layer 4: Predicted route – classifier only (orange dashed)
  Layer 5: Predicted route – HMM Viterbi (red solid)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from typing import Optional, List

try:
    import folium
    from folium import LayerControl, FeatureGroup, CircleMarker, PolyLine, GeoJson
    from folium.plugins import MeasureControl
except ImportError:
    raise ImportError("Run: pip install folium")

from src.data.loader import TripData, load_trip, get_available_trip_ids
from src.features.spatial import LocalMetricProjector
from src.features.alignment import align_route_to_observations


def build_trip_map(
    trip: TripData,
    clf_pred_arcs: Optional[np.ndarray] = None,
    hmm_pred_arcs: Optional[np.ndarray] = None,
    title: str = "",
) -> folium.Map:
    """
    Build a 4-layer interactive Folium map for a single trip.

    Args:
        trip:           TripData with track, nodes, arcs (and optionally route).
        clf_pred_arcs:  (N,) array of classifier-predicted arc IDs per observation.
        hmm_pred_arcs:  (N,) array of Viterbi-decoded arc IDs per observation.
        title:          Map title shown in a tile layer.

    Returns:
        Folium Map object.
    """
    track  = trip.track
    nodes  = trip.nodes
    arcs   = trip.arcs

    # Center map on GPS track centroid
    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)

    # ── Layer 1: Raw GPS Observations ───────────────────────────
    gps_group = FeatureGroup(name="🔵 Raw GPS Observations", show=True)
    timestamps = track["t"].values
    for i in range(len(track)):
        t_sec = int(timestamps[i])
        CircleMarker(
            location=[track["lat"].iloc[i], track["lon"].iloc[i]],
            radius=3,
            color="#2196F3",
            fill=True,
            fill_color="#2196F3",
            fill_opacity=0.7,
            tooltip=f"Obs #{i}  t={t_sec}s",
        ).add_to(gps_group)
    gps_group.add_to(m)

    # ── Layer 2: Road Network ────────────────────────────────────
    road_group = FeatureGroup(name="⬜ Road Network", show=False)
    for arc_id, (u, v) in enumerate(zip(arcs["u"].values, arcs["v"].values)):
        u_lat, u_lon = nodes["lat"].iloc[u], nodes["lon"].iloc[u]
        v_lat, v_lon = nodes["lat"].iloc[v], nodes["lon"].iloc[v]
        PolyLine(
            locations=[[u_lat, u_lon], [v_lat, v_lon]],
            color="#9E9E9E",
            weight=1.5,
            opacity=0.5,
            tooltip=f"arc {arc_id}",
        ).add_to(road_group)
    road_group.add_to(m)

    # ── Layer 3: Ground-Truth Route ──────────────────────────────
    if trip.route is not None and len(trip.route) > 0:
        gt_group = FeatureGroup(name="🟢 Ground-Truth Route", show=True)
        gt_coords = []
        for arc_id in trip.route:
            u = int(arcs["u"].iloc[arc_id])
            v = int(arcs["v"].iloc[arc_id])
            gt_coords.append([nodes["lat"].iloc[u], nodes["lon"].iloc[u]])
        # Add endpoint of last arc
        last_arc = int(trip.route[-1])
        v_last = int(arcs["v"].iloc[last_arc])
        gt_coords.append([nodes["lat"].iloc[v_last], nodes["lon"].iloc[v_last]])

        PolyLine(
            locations=gt_coords,
            color="#4CAF50",
            weight=5,
            opacity=0.85,
            tooltip="Ground Truth Route",
        ).add_to(gt_group)
        gt_group.add_to(m)

    # ── Layer 4: Classifier-Only Predicted Route ─────────────────
    if clf_pred_arcs is not None and len(clf_pred_arcs) > 0:
        clf_group = FeatureGroup(name="🟠 Predicted Route (Classifier Only)", show=True)
        clf_coords = _arcs_to_coords(clf_pred_arcs, arcs, nodes)
        PolyLine(
            locations=clf_coords,
            color="#FF9800",
            weight=3,
            opacity=0.80,
            dash_array="6 4",
            tooltip="Classifier-Only Prediction",
        ).add_to(clf_group)
        clf_group.add_to(m)

    # ── Layer 5: HMM Viterbi Predicted Route ─────────────────────
    if hmm_pred_arcs is not None and len(hmm_pred_arcs) > 0:
        hmm_group = FeatureGroup(name="🔴 Predicted Route (HMM + Viterbi)", show=True)
        hmm_coords = _arcs_to_coords(hmm_pred_arcs, arcs, nodes)
        PolyLine(
            locations=hmm_coords,
            color="#F44336",
            weight=4,
            opacity=0.90,
            tooltip="HMM + Viterbi Prediction",
        ).add_to(hmm_group)
        hmm_group.add_to(m)

    # ── Legend & Controls ─────────────────────────────────────────
    LayerControl(collapsed=False).add_to(m)
    MeasureControl(primary_length_unit="meters").add_to(m)

    # Inline HTML legend
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif; font-size: 13px;">
        <b>PathSnap — {title}</b><br><br>
        <span style="color:#2196F3;">●</span> Raw GPS Observations<br>
        <span style="color:#9E9E9E;">─</span> Road Network<br>
        <span style="color:#4CAF50;">━</span> Ground-Truth Route<br>
        <span style="color:#FF9800;">╌</span> Classifier-Only Prediction<br>
        <span style="color:#F44336;">━</span> HMM + Viterbi Prediction<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def _arcs_to_coords(
    arc_seq: np.ndarray,
    arcs: pd.DataFrame,
    nodes: pd.DataFrame,
) -> List[List[float]]:
    """Convert a sequence of arc IDs to [[lat, lon], ...] coordinates."""
    coords = []
    prev_arc = None
    for arc_id in arc_seq:
        arc_id = int(arc_id)
        if arc_id == prev_arc:
            continue  # skip consecutive duplicates
        u = int(arcs["u"].iloc[arc_id])
        lat = float(nodes["lat"].iloc[u])
        lon = float(nodes["lon"].iloc[u])
        coords.append([lat, lon])
        prev_arc = arc_id
    # Add endpoint of last arc
    if len(arc_seq) > 0:
        last_arc = int(arc_seq[-1])
        v = int(arcs["v"].iloc[last_arc])
        coords.append([float(nodes["lat"].iloc[v]), float(nodes["lon"].iloc[v])])
    return coords


def generate_all_trip_maps(
    partition_dir: str,
    out_dir: str,
    model_path: Optional[str] = None,
    viterbi_results_csv: Optional[str] = None,
    max_trips: Optional[int] = None,
) -> List[str]:
    """
    Generate Folium HTML maps for all trips in a partition.

    Args:
        partition_dir:       Path to trip partition (e.g. "Student folder/student_test").
        out_dir:             Directory to save .html files.
        model_path:          If provided, run classifier predictions to overlay.
        viterbi_results_csv: Path to CSV with 'trip_id' + decoded arcs columns (optional).
        max_trips:           Limit number of trips processed.

    Returns:
        List of saved HTML file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    trip_ids = get_available_trip_ids(partition_dir)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    model = None
    if model_path and os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

    saved_paths = []
    print(f"Generating maps for {len(trip_ids)} trips...")

    for i, trip_id in enumerate(trip_ids):
        trip_path = os.path.join(partition_dir, trip_id)
        trip      = load_trip(trip_path, trip_id, load_route=True)

        clf_pred = None
        hmm_pred = None

        if model is not None:
            # Run classifier predictions
            try:
                from src.features.build_features import extract_features_for_trip, FEATURE_COLUMNS
                feat_df, _ = extract_features_for_trip(trip, k_candidates=8, is_training=False)
                X = feat_df[FEATURE_COLUMNS].values
                probs = model.predict_proba(X)[:, 1]
                feat_df["pred_score"] = probs
                clf_pred = (
                    feat_df.sort_values(["obs_idx", "pred_score"], ascending=[True, False])
                    .groupby("obs_idx")["arc_id"]
                    .first()
                    .values
                )

                # Try Viterbi
                try:
                    from Phase_2.hmm.viterbi import viterbi_decode_trip
                    hmm_pred, _, _ = viterbi_decode_trip(trip, model)
                except Exception as e:
                    print(f"  Viterbi skipped for {trip_id}: {e}")
            except Exception as e:
                print(f"  Prediction skipped for {trip_id}: {e}")

        folium_map = build_trip_map(
            trip, clf_pred, hmm_pred, title=f"Trip {trip_id}"
        )
        out_path = os.path.join(out_dir, f"trip_{trip_id}_map.html")
        folium_map.save(out_path)
        saved_paths.append(out_path)
        print(f"  [{i+1}/{len(trip_ids)}] Saved {out_path}")

    return saved_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="Student folder/student_test")
    parser.add_argument("--out",       default="artifacts/plots/folium_maps")
    parser.add_argument("--model",     default="artifacts/models/HistGradientBoosting_tuned.joblib")
    parser.add_argument("--max-trips", type=int, default=None)
    args = parser.parse_args()

    paths = generate_all_trip_maps(
        args.partition, args.out,
        model_path=args.model,
        max_trips=args.max_trips,
    )
    print(f"\nGenerated {len(paths)} map(s).")
