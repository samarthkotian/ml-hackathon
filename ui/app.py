"""
Laptop 2 — PathSnap Evaluator User Interface (Streamlit) — Team SI No 107.

Connects over HTTP to Laptop 1's FastAPI server, fetches the 10 reserved
evaluation trips, sends the selected trip_id for live route reconstruction
and server-side evaluation, and renders the interactive Folium map and metrics.

Run on Laptop 2:
    streamlit run ui/app.py
"""
import json
import os
import sys
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui_state import select_evaluation, request_evaluation

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard — SI No 107",
    page_icon="🗺️",
    layout="wide",
)

# Custom styling for crisp metric cards and header
st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
    .main-banner h1 {
        color: white;
        margin: 0;
        font-size: 26px;
    }
    .main-banner p {
        color: #cbd5e1;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-banner">
        <h1>🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction</h1>
        <p><b>Team SI No 107</b> — Saatvik Gupta (PES1UG24CS395) • Sagar A Walikar (PES1UG24CS396) • Samarth Kotian (PES1UG24CS406) • Samarth M H (PES1UG24CS407)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Laptop 1 API Configuration
with st.sidebar:
    st.header("🔌 Laptop 1 API Connection")
    default_api = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")
    api_url = st.text_input(
        "FastAPI Server URL (Laptop 1)",
        value=default_api,
        help="Enter http://<LAPTOP_1_LAN_IP>:8000 during Two-Laptop Evaluation",
    ).rstrip("/")

    if st.button("🔄 Check Server Health", use_container_width=True):
        st.session_state["force_refresh"] = True

    health_data = None
    try:
        resp = requests.get(f"{api_url}/health", timeout=4)
        if resp.status_code == 200:
            health_data = resp.json()
            st.success(f"Connected to Laptop 1 API ({api_url})")
            st.caption(f"**Active Model:** `{health_data.get('selected_model')}`")
            st.caption(f"**Available Trips:** `{health_data.get('available_trips_count')}`")
        else:
            st.error(f"API returned HTTP {resp.status_code}")
    except Exception as exc:
        st.error(f"Cannot reach API at `{api_url}`.\n\nEnsure Laptop 1 is running:\n`uvicorn api.main:app --host 0.0.0.0 --port 8000`")

    st.divider()
    st.subheader("📋 Two-Laptop Architecture")
    st.markdown(
        """
        - **Laptop 1 (API & Model Server):** Hosts Bayesian-Tuned LightGBM classifier, Road-Graph HMM/Viterbi decoder, and internal `.route` reference evaluator.
        - **Laptop 2 (Evaluator UI):** Displays evaluation trips, dispatches `trip_id` over HTTP, and visualises reconstructed routes.
        """
    )

# Fetch available evaluation trips from Laptop 1 API
trips_list = []
try:
    t_resp = requests.get(f"{api_url}/trips", timeout=5)
    if t_resp.status_code == 200:
        trips_list = t_resp.json().get("trips", [])
except Exception:
    pass

if not trips_list:
    st.warning("Waiting for connection to Laptop 1 FastAPI server to load evaluation trips...")
    st.stop()

# Filter controls & Trip selector
col_sel, col_part, col_btn = st.columns([2.5, 1.5, 1.5])

partitions_available = sorted(list({t["partition"] for t in trips_list}))
part_options = ["evaluator_test"] + [p for p in partitions_available if p != "evaluator_test"] + ["All"] if "evaluator_test" in partitions_available else ["All"] + partitions_available
with col_part:
    selected_partition = st.selectbox(
        "Dataset Partition",
        options=part_options,
        index=0,
    )

filtered_trips = [
    t for t in trips_list if selected_partition == "All" or t["partition"] == selected_partition
]

with col_sel:
    selected_label = st.selectbox(
        "Select Evaluation Trip (10 Reserved Trips)",
        options=[t["label"] for t in filtered_trips],
        index=0,
    )

selected_trip_obj = next(t for t in filtered_trips if t["label"] == selected_label)

with col_btn:
    st.write("")
    st.write("")
    run_eval = st.button(
        "🚀 Reconstruct Route via API",
        type="primary",
        use_container_width=True,
    )

evaluation_key = (api_url, selected_trip_obj["trip_id"], selected_trip_obj["partition"])
selection_changed = select_evaluation(st.session_state, evaluation_key)
if run_eval or selection_changed:
    with st.spinner(f"Requesting live route reconstruction for Trip `{selected_trip_obj['trip_id']}` from `{api_url}/evaluate`..."):
        try:
            payload = {
                "trip_id": selected_trip_obj["trip_id"],
                "partition": selected_trip_obj["partition"],
                "include_map_html": True,
            }
            request_evaluation(
                st.session_state, evaluation_key,
                lambda: requests.post(f"{api_url}/evaluate", json=payload, timeout=30),
            )
        except Exception as exc:
            st.error(f"Request to Laptop 1 API failed: {exc}")

res = st.session_state.get("last_eval_result")
if res:
    m = res["metrics"]
    st.subheader(
        f"📊 Live Evaluation Results — Trip `{res['trip_id']}` (`{res['partition']}`)"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    if m.get("has_ground_truth"):
        k1.metric(
            "Top-1 Accuracy (HMM)",
            f"{m['hmm_top1_accuracy']*100:.2f}%",
            delta=f"{m['top1_improvement_pct']:+.2f}% vs Clf-Only",
        )
        k2.metric(
            "Candidate Macro F1",
            f"{m['macro_f1']:.4f}",
            delta=f"Miss Rate: {m['candidate_miss_rate']*100:.2f}%",
            delta_color="off",
        )
        k3.metric(
            "Route Continuity (HMM)",
            f"{m['route_continuity']*100:.2f}%",
            delta=f"{(m['route_continuity'] - m['clf_only_continuity'])*100:+.2f}% vs Clf-Only",
        )
        k4.metric(
            "Sequence Route F1",
            f"{m['route_seq_f1']:.4f}",
            delta=f"Jaccard: {m['route_jaccard']:.4f}",
            delta_color="off",
        )
    else:
        k1.metric("Top-1 Accuracy", "Server-Internal")
        k2.metric("Candidate Macro F1", "Server-Internal")
        k3.metric(
            "Route Continuity (HMM)",
            f"{m['route_continuity']*100:.2f}%",
            delta=f"{(m['route_continuity'] - m['clf_only_continuity'])*100:+.2f}% vs Clf-Only",
        )
        k4.metric("Strict 1-Hop Continuity", f"{m['strict_1hop_continuity']*100:.2f}%")

    k5.metric(
        "GPS Pts → Route Arcs",
        f"{m['n_observations']} → {m['n_reconstructed_route_arcs']}",
    )
    k6.metric("API Inference Time", f"{m['inference_ms']:.1f} ms")

    st.markdown("### 🗺️ Interactive Folium Route Reconstruction Map")
    st.caption(
        "Use the layer control in the top-right corner of the map to toggle Raw GPS Observations (Blue), "
        "Classifier-Only Predictions (Orange Dashed), and "
        "PathSnap HMM/Viterbi Reconstructed Route (Red)."
    )
    if res.get("folium_map_html"):
        components.html(res["folium_map_html"], height=600, scrolling=False)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        route_str = "\n".join(str(x) for x in res["reconstructed_route_arc_ids"])
        st.download_button(
            label=f"⬇️ Download Predicted {res['trip_id']}.route ({len(res['reconstructed_route_arc_ids'])} arcs)",
            data=route_str,
            file_name=f"{res['trip_id']}.route",
            mime="text/plain",
            use_container_width=True,
        )
    with col_d2:
        obs_str = "\n".join(str(x) for x in res["observation_predicted_arc_ids"])
        st.download_button(
            label=f"⬇️ Download Observation-Level Arc IDs ({len(res['observation_predicted_arc_ids'])} pts)",
            data=obs_str,
            file_name=f"{res['trip_id']}_obs_arcs.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("🔍 Inspect Raw JSON API Response from Laptop 1"):
        st.json({k: v for k, v in res.items() if k != "folium_map_html"})
