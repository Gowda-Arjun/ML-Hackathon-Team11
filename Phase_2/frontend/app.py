"""Streamlit Evaluator UI – Laptop 2 (Client).

Connects to the FastAPI server on Laptop 1 and renders:
  - Trip selector
  - "Run Map-Matching" button
  - Scorecard: Accuracy, Continuity, Latency
  - Embedded Folium map (raw GPS + road network + predicted routes)

Usage (on Laptop 2):
  pip install streamlit requests folium streamlit-folium
  streamlit run Phase_2/frontend/app.py
"""

import os
import sys

try:
    import streamlit as st
    import requests
    import folium
    from streamlit_folium import st_folium
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install streamlit requests folium streamlit-folium"
    )

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_SERVER = os.environ.get("PATHSNAP_SERVER", "http://localhost:8000")

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PathSnap — Evaluator UI",
    page_icon="🗺️",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid;
        margin-bottom: 12px;
    }
    .metric-label { font-size: 0.85rem; color: #666; font-weight: 600; }
    .metric-value { font-size: 2rem; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 0.8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/map-marker.png", width=60)
    st.title("PathSnap")
    st.caption("Distributed Map-Matching & Route Reconstruction")
    st.divider()
    server_url = st.text_input("🖥️ Laptop 1 Server URL", value=DEFAULT_SERVER)

    # Health check
    if st.button("🔄 Check Server Health"):
        try:
            resp = requests.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                h = resp.json()
                st.success(f"✅ Server OK — {h['n_eval_trips']} trips available")
                st.caption(f"Model loaded: {h['model_loaded']}")
            else:
                st.error(f"Server returned {resp.status_code}")
        except Exception as e:
            st.error(f"Cannot reach server: {e}")

    st.divider()
    st.caption("📡 Sends requests to Laptop 1 over LAN.\nGround truth is never transmitted.")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🗺️ PathSnap Evaluator")
st.markdown(
    "Select an evaluation trip, run map-matching inference on **Laptop 1**, "
    "and visualise the reconstructed route here."
)

# Fetch available trips
@st.cache_data(ttl=30, show_spinner=False)
def fetch_trips(server_url: str):
    try:
        resp = requests.get(f"{server_url}/trips", timeout=5)
        resp.raise_for_status()
        return resp.json()["trip_ids"]
    except Exception:
        return []

trip_ids = fetch_trips(server_url)

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    if trip_ids:
        selected_trip = st.selectbox("📍 Select Evaluation Trip", options=trip_ids)
    else:
        st.warning("Could not fetch trips. Check server URL in sidebar.")
        selected_trip = None

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run Map-Matching", type="primary", disabled=(selected_trip is None))

# ─── Inference & Display ──────────────────────────────────────────────────────
if run_btn and selected_trip:
    with st.spinner(f"Running inference for trip {selected_trip}..."):
        try:
            resp = requests.get(
                f"{server_url}/predict/{selected_trip}", timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    # ── Scorecard ────────────────────────────────────────────────
    st.subheader("📊 Results Scorecard")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "🎯 Classifier Top-1 Accuracy",
            f"{data['clf_top1_accuracy']*100:.2f}%",
        )
    with c2:
        delta_acc = data["hmm_top1_accuracy"] - data["clf_top1_accuracy"]
        st.metric(
            "🤖 HMM Top-1 Accuracy",
            f"{data['hmm_top1_accuracy']*100:.2f}%",
            delta=f"{delta_acc*100:+.2f}%",
        )
    with c3:
        st.metric(
            "🔗 Clf Route Continuity",
            f"{data['clf_route_continuity']*100:.2f}%",
        )
    with c4:
        delta_cont = data["hmm_route_continuity"] - data["clf_route_continuity"]
        st.metric(
            "🔗 HMM Route Continuity",
            f"{data['hmm_route_continuity']*100:.2f}%",
            delta=f"{delta_cont*100:+.2f}%",
        )
    with c5:
        st.metric("⏱️ Latency", f"{data['inference_time_ms']:.0f} ms")

    st.caption(
        f"Trip `{data['trip_id']}` — {data['n_observations']} GPS observations — "
        f"Candidate miss rate: {data['candidate_miss_rate']*100:.2f}%"
    )

    # ── Build Folium Map ─────────────────────────────────────────
    st.subheader("🗺️ Interactive Route Map")

    gps_coords = data["raw_gps_coords"]
    clf_coords = data["clf_route_coords"]
    hmm_coords = data["hmm_route_coords"]
    road_arcs  = data.get("road_arcs_sample", [])

    if gps_coords:
        center = gps_coords[len(gps_coords) // 2]
    else:
        center = [0, 0]

    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron")

    # Road network sample
    road_grp = folium.FeatureGroup(name="⬜ Road Network", show=False)
    for arc in road_arcs:
        folium.PolyLine(
            [[arc["u_lat"], arc["u_lon"]], [arc["v_lat"], arc["v_lon"]]],
            color="#9E9E9E", weight=1.5, opacity=0.5,
        ).add_to(road_grp)
    road_grp.add_to(m)

    # Raw GPS
    gps_grp = folium.FeatureGroup(name="🔵 Raw GPS Observations", show=True)
    for idx, (lat, lon) in enumerate(gps_coords[::max(1, len(gps_coords) // 500)]):
        folium.CircleMarker(
            [lat, lon], radius=3, color="#2196F3",
            fill=True, fill_color="#2196F3", fill_opacity=0.7,
            tooltip=f"obs #{idx}",
        ).add_to(gps_grp)
    gps_grp.add_to(m)

    # Classifier route
    if clf_coords:
        clf_grp = folium.FeatureGroup(name="🟠 Classifier-Only Route", show=True)
        folium.PolyLine(clf_coords, color="#FF9800", weight=3,
                        opacity=0.8, dash_array="6 4",
                        tooltip="Classifier-Only").add_to(clf_grp)
        clf_grp.add_to(m)

    # HMM route
    if hmm_coords:
        hmm_grp = folium.FeatureGroup(name="🔴 HMM + Viterbi Route", show=True)
        folium.PolyLine(hmm_coords, color="#F44336", weight=4,
                        opacity=0.9, tooltip="HMM + Viterbi").add_to(hmm_grp)
        hmm_grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);font-size:12px;">
        <b>PathSnap Legend</b><br>
        <span style="color:#9E9E9E;">─</span> Road Network<br>
        <span style="color:#2196F3;">●</span> Raw GPS<br>
        <span style="color:#FF9800;">╌</span> Classifier Only<br>
        <span style="color:#F44336;">━</span> HMM + Viterbi<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=560, returned_objects=[])

    # ── Raw JSON ─────────────────────────────────────────────────
    with st.expander("📋 Raw API Response (JSON)"):
        display_data = {k: v for k, v in data.items()
                        if k not in ("raw_gps_coords", "clf_route_coords",
                                     "hmm_route_coords", "road_arcs_sample")}
        st.json(display_data)

else:
    st.info("👆 Select a trip and click **Run Map-Matching** to start inference.")
