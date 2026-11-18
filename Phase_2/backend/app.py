"""FastAPI Backend Server for PathSnap – Laptop 1.

Endpoints:
  GET  /health           – Health check
  GET  /trips            – List available evaluation trip IDs
  POST /predict/{trip_id} – Full inference pipeline + metric evaluation

Ground-truth labels are loaded server-side only and NEVER returned to the client.

Run:
  cd /path/to/mlhack
  uvicorn Phase_2.backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn")

import numpy as np
import pandas as pd
import joblib

from src.data.loader import TripData, load_trip, get_available_trip_ids
from src.features.spatial import LocalMetricProjector
from src.features.build_features import extract_features_for_trip, FEATURE_COLUMNS
from src.features.alignment import align_route_to_observations
from src.models.metrics import compute_observation_metrics
from Phase_2.hmm.viterbi import viterbi_decode_trip, _route_continuity


# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("pathsnap.backend")

# ─── Configuration ────────────────────────────────────────────────────────────
# Change these to match your machine's layout
EVAL_PARTITION_DIR = os.environ.get(
    "PATHSNAP_EVAL_DIR",
    "Student folder/student_test",
)
MODEL_PATH = os.environ.get(
    "PATHSNAP_MODEL",
    "artifacts/models/HistGradientBoosting_tuned.joblib",
)
# Fall back to untuned model if tuned doesn't exist yet
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "artifacts/models/HistGradientBoosting.joblib"

# ─── App & State ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Route Reconstruction — Laptop 1 Backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (loaded at startup)
_model = None
_trip_ids: List[str] = []


@app.on_event("startup")
async def startup_event():
    global _model, _trip_ids
    logger.info("PathSnap Backend starting up...")

    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. /predict will return 503.")

    _trip_ids = get_available_trip_ids(EVAL_PARTITION_DIR)
    logger.info(f"Available evaluation trips ({len(_trip_ids)}): {_trip_ids}")


# ─── Response Schemas ─────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_eval_trips: int
    timestamp: str


class TripsResponse(BaseModel):
    trip_ids: List[str]
    partition_dir: str


class PredictResponse(BaseModel):
    trip_id: str
    # Metrics (ground truth computed server-side, never returned in full)
    clf_top1_accuracy: float
    hmm_top1_accuracy: float
    clf_route_continuity: float
    hmm_route_continuity: float
    # Route GeoJSON-style coordinates for the client to render
    raw_gps_coords: List[List[float]]          # [[lat, lon], ...]
    clf_route_coords: List[List[float]]         # [[lat, lon], ...]
    hmm_route_coords: List[List[float]]         # [[lat, lon], ...]
    road_arcs_sample: List[Dict[str, Any]]      # sample of road arcs for map rendering
    # Performance
    inference_time_ms: float
    n_observations: int
    candidate_miss_rate: float


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        n_eval_trips=len(_trip_ids),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/trips", response_model=TripsResponse)
def list_trips():
    return TripsResponse(trip_ids=_trip_ids, partition_dir=EVAL_PARTITION_DIR)


@app.get("/predict/{trip_id}", response_model=PredictResponse)
def predict(trip_id: str):
    if _model is None:
        raise HTTPException(503, detail="Model not loaded. Run Phase 4 first.")

    if trip_id not in _trip_ids:
        raise HTTPException(404, detail=f"Trip '{trip_id}' not found in {EVAL_PARTITION_DIR}.")

    logger.info(f"Received predict request for trip_id={trip_id}")
    t_start = time.time()

    # ── Load trip ──────────────────────────────────────────────
    trip_path = os.path.join(EVAL_PARTITION_DIR, trip_id)
    trip      = load_trip(trip_path, trip_id, load_route=True)

    n_obs = len(trip.track)
    center_lon = float(trip.track["lon"].mean())
    center_lat = float(trip.track["lat"].mean())
    projector  = LocalMetricProjector(center_lon, center_lat)

    # ── Ground-truth alignment (server-side only) ──────────────
    ref_arcs, _ = align_route_to_observations(trip, projector)

    # ── Classifier-only inference ──────────────────────────────
    feat_df, meta = extract_features_for_trip(trip, k_candidates=8, is_training=False)
    X = feat_df[FEATURE_COLUMNS].values
    probs = _model.predict_proba(X)[:, 1]
    feat_df["pred_score"] = probs

    top1_df = (
        feat_df.sort_values(["obs_idx", "pred_score"], ascending=[True, False])
        .groupby("obs_idx")
        .first()
        .reset_index()
    )
    clf_pred_arcs = top1_df["arc_id"].values

    clf_top1_acc  = float(np.mean(clf_pred_arcs == ref_arcs[:len(clf_pred_arcs)]))
    clf_continuity = _route_continuity(clf_pred_arcs, trip.arcs)

    # ── HMM + Viterbi ─────────────────────────────────────────
    try:
        hmm_pred_arcs, _, _ = viterbi_decode_trip(trip, _model, k_candidates=8)
        hmm_top1_acc    = float(np.mean(hmm_pred_arcs == ref_arcs[:len(hmm_pred_arcs)]))
        hmm_continuity  = _route_continuity(hmm_pred_arcs, trip.arcs)
    except Exception as exc:
        logger.warning(f"Viterbi failed for trip {trip_id}: {exc}. Falling back to clf.")
        hmm_pred_arcs  = clf_pred_arcs
        hmm_top1_acc   = clf_top1_acc
        hmm_continuity = clf_continuity

    inference_ms = round((time.time() - t_start) * 1000, 1)
    logger.info(
        f"Trip {trip_id}: clf_top1={clf_top1_acc:.3f} hmm_top1={hmm_top1_acc:.3f} "
        f"clf_cont={clf_continuity:.3f} hmm_cont={hmm_continuity:.3f} "
        f"latency={inference_ms}ms"
    )

    # ── Build coordinate payloads for client ───────────────────
    raw_gps_coords = [
        [float(trip.track["lat"].iloc[i]), float(trip.track["lon"].iloc[i])]
        for i in range(n_obs)
    ]
    clf_route_coords = _arcs_to_coords(clf_pred_arcs, trip.arcs, trip.nodes)
    hmm_route_coords = _arcs_to_coords(hmm_pred_arcs, trip.arcs, trip.nodes)

    # Sample up to 500 road arcs for the client to render
    n_arcs   = len(trip.arcs)
    sample_step = max(1, n_arcs // 500)
    road_arcs_sample = []
    for arc_id in range(0, n_arcs, sample_step):
        u = int(trip.arcs["u"].iloc[arc_id])
        v = int(trip.arcs["v"].iloc[arc_id])
        road_arcs_sample.append({
            "arc_id": arc_id,
            "u_lat": float(trip.nodes["lat"].iloc[u]),
            "u_lon": float(trip.nodes["lon"].iloc[u]),
            "v_lat": float(trip.nodes["lat"].iloc[v]),
            "v_lon": float(trip.nodes["lon"].iloc[v]),
        })

    return PredictResponse(
        trip_id=trip_id,
        clf_top1_accuracy=round(clf_top1_acc, 4),
        hmm_top1_accuracy=round(hmm_top1_acc, 4),
        clf_route_continuity=round(clf_continuity, 4),
        hmm_route_continuity=round(hmm_continuity, 4),
        raw_gps_coords=raw_gps_coords,
        clf_route_coords=clf_route_coords,
        hmm_route_coords=hmm_route_coords,
        road_arcs_sample=road_arcs_sample,
        inference_time_ms=inference_ms,
        n_observations=n_obs,
        candidate_miss_rate=round(meta.get("miss_rate", 0.0), 4),
    )


def _arcs_to_coords(arc_seq, arcs_df, nodes_df):
    coords = []
    prev   = None
    for arc_id in arc_seq:
        arc_id = int(arc_id)
        if arc_id == prev:
            continue
        u = int(arcs_df["u"].iloc[arc_id])
        coords.append([float(nodes_df["lat"].iloc[u]), float(nodes_df["lon"].iloc[u])])
        prev = arc_id
    if len(arc_seq) > 0:
        v = int(arcs_df["v"].iloc[int(arc_seq[-1])])
        coords.append([float(nodes_df["lat"].iloc[v]), float(nodes_df["lon"].iloc[v])])
    return coords
