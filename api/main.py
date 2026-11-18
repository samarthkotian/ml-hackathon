"""
Laptop 1 — PathSnap FastAPI Server & Server-Side Evaluation Engine (Team SI No 107).

Responsibilities:
- Hosts the trained Bayesian-Tuned LightGBM classifier and HMM/Viterbi decoder
- Loads road-network (.nodes, .arcs) and raw GPS trajectories (.track)
- Keeps reference ground-truth (.route) strictly server-side for evaluation only
- Logs detailed live inference telemetry to CMD console for evaluator verification

Run on Laptop 1:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from datetime import datetime
import os
import time
from typing import Any, Dict, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DATA_ROOT,
    EVALUATOR_TEST_DIR,
    STUDENT_TEST_DIR,
    TEAM_INFO,
    TRAIN_DIR,
)
from src.data_loader import TripData, load_trip_from_dir
from src.paths import TRIP_ID_PATTERN, safe_child, validate_trip_id
from src.visualization import generate_folium_map_for_trip
from predict import load_inference_artifacts, predict_single_trip

app = FastAPI(
    title="PathSnap Distributed Map-Matching & Route Reconstruction API",
    description="Laptop 1 Model & Evaluation Server — Team SI No 107 (PES University)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache loaded on startup
_MODEL = None
_HMM_CFG = None
_METADATA: Dict[str, Any] = {}


def _get_artifacts():
    global _MODEL, _HMM_CFG, _METADATA
    if _MODEL is None:
        _MODEL, _HMM_CFG, _METADATA = load_inference_artifacts()
    return _MODEL, _HMM_CFG, _METADATA


def _discover_evaluation_trips() -> List[Dict[str, str]]:
    """
    Discover available evaluation trips on Laptop 1.
    Checks `evaluator_test` first (for the 10 reserved evaluator trips) and
    `student_test` (10 development evaluation trips) so both partitions are
    seamlessly accessible from Laptop 2's UI.
    """
    trips_info = []
    search_dirs = [
        ("evaluator_test", EVALUATOR_TEST_DIR),
        ("student_test", STUDENT_TEST_DIR),
        ("evaluator_test", os.path.join(os.path.dirname(DATA_ROOT), "evaluator_test")),
    ]
    seen = set()
    for part_name, dpath in search_dirs:
        if os.path.exists(dpath):
            for tid in sorted(os.listdir(dpath)):
                try:
                    validate_trip_id(tid)
                    tdir = safe_child(dpath, tid)
                    track_file = safe_child(tdir, f"{tid}.track")
                except ValueError:
                    continue
                if os.path.isdir(tdir) and os.path.exists(track_file) and (part_name, tid) not in seen:
                    seen.add((part_name, tid))
                    trips_info.append(
                        {
                            "trip_id": tid,
                            "partition": part_name,
                            "trip_dir": tdir,
                            "label": f"Trip {tid} ({part_name})",
                        }
                    )
    return trips_info


class EvaluateRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN, min_length=1, max_length=64)
    partition: Optional[Literal["train", "student_test", "evaluator_test"]] = None
    include_map_html: bool = True


@app.on_event("startup")
def startup_event() -> None:
    try:
        _, _, meta = _get_artifacts()
        trips = _discover_evaluation_trips()
        print("\n" + "=" * 82, flush=True)
        print(" [LAPTOP 1 — PATHSNAP FASTAPI SERVER ONLINE]", flush=True)
        print(f" Team: {TEAM_INFO['team_id']} | Model: {meta.get('selected_model_name', 'Bayesian-Tuned LightGBM')}", flush=True)
        print(f" Available Evaluation Trips Discovered: {len(trips)}", flush=True)
        print("=" * 82 + "\n", flush=True)
    except Exception as exc:
        print(f"[STARTUP WARNING] Model artifacts not yet loaded: {exc}", flush=True)


@app.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    _, _, meta = _get_artifacts()
    trips = _discover_evaluation_trips()
    client_ip = request.client.host if request.client else "unknown"
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] GET /health from {client_ip} -> OK ({len(trips)} trips ready)",
        flush=True,
    )
    return {
        "status": "online",
        "team": TEAM_INFO,
        "selected_model": meta.get("selected_model_name", "Bayesian-Tuned LightGBM"),
        "available_trips_count": len(trips),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/trips")
def list_trips(request: Request) -> Dict[str, Any]:
    trips = _discover_evaluation_trips()
    client_ip = request.client.host if request.client else "unknown"
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] GET /trips from {client_ip} -> Returning {len(trips)} evaluation trip options",
        flush=True,
    )
    # Do not expose internal file paths or ground-truth route data to the client UI
    public_trips = [
        {"trip_id": t["trip_id"], "partition": t["partition"], "label": t["label"]}
        for t in trips
    ]
    return {"trips": public_trips, "count": len(public_trips)}


@app.post("/evaluate")
def evaluate_trip_endpoint(req: EvaluateRequest, request: Request) -> Dict[str, Any]:
    t_req = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "-" * 82, flush=True)
    print(f"[{now_str}] INCOMING INFERENCE REQUEST: POST /evaluate", flush=True)
    print(f"  Client IP (Laptop 2): {client_ip} | Requested trip_id: {req.trip_id} (partition={req.partition or 'auto'})", flush=True)

    trips = _discover_evaluation_trips()
    matched = [
        t
        for t in trips
        if t["trip_id"] == req.trip_id
        and (req.partition is None or t["partition"] == req.partition)
    ]
    if not matched and req.partition == "train":
        # Training trips are available only when explicitly requested.
        try:
            tr_dir = safe_child(TRAIN_DIR, req.trip_id)
            track_file = safe_child(tr_dir, f"{req.trip_id}.track")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if os.path.exists(track_file):
            matched = [{"trip_id": req.trip_id, "partition": "train", "trip_dir": tr_dir}]
    if not matched:
        raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found in the requested partition.")

    target = matched[0]
    model, hmm_cfg, meta = _get_artifacts()

    # Load trip on server; note that prediction pipeline uses ONLY .track, .nodes, .arcs
    trip_obj: TripData = load_trip_from_dir(
        target["trip_dir"], target["trip_id"], partition=target["partition"]
    )
    print(
        f"  [Step 1] Loaded Trip {trip_obj.trip_id}: {len(trip_obj.track)} GPS pts, "
        f"{len(trip_obj.nodes)} road nodes, {len(trip_obj.arcs)} directed arcs",
        flush=True,
    )

    pred_result = predict_single_trip(
        trip_obj, model, hmm_cfg, save_outputs=True, generate_map=False
    )
    metrics = pred_result["metrics"]

    print(
        f"  [Step 2] Pipeline Executed in {metrics['inference_ms']:.1f} ms "
        f"(Candidate Gen + 25 Features + {meta.get('selected_model_name', 'LightGBM')} + Log-Viterbi)",
        flush=True,
    )
    if metrics.get("has_ground_truth"):
        print(
            f"  [Step 3] Server-Side Evaluation vs Internal Reference Route:\n"
            f"           -> Observation-Level Top-1 Accuracy (HMM/Viterbi): {metrics['hmm_top1_accuracy']*100:.2f}%\n"
            f"           -> Observation-Level Top-1 Accuracy (Clf-Only):    {metrics['clf_top1_accuracy']*100:.2f}% (+{metrics['top1_improvement_pct']:.2f}%)\n"
            f"           -> Candidate Macro F1-Score:                       {metrics['macro_f1']:.4f}\n"
            f"           -> Topological Route Continuity (HMM/Viterbi):     {metrics['route_continuity']*100:.2f}% (Clf-Only: {metrics['clf_only_continuity']*100:.2f}%)",
            flush=True,
        )

    map_html_str = ""
    if req.include_map_html:
        import numpy as np
        fmap = generate_folium_map_for_trip(
            trip_obj,
            clf_only_arcs=np.array(pred_result["clf_pred_arcs"], dtype=int),
            hmm_pred_arcs=np.array(pred_result["hmm_pred_arcs"], dtype=int),
            hmm_reconstructed_route=pred_result["reconstructed_route"],
            out_html_path=None,
            include_reference=False,
        )
        map_html_str = fmap.get_root().render()

    total_ms = (time.perf_counter() - t_req) * 1000.0
    print(f"  [Step 4] Returning HTTP 200 OK to {client_ip} (Total Roundtrip: {total_ms:.1f} ms)", flush=True)
    print("-" * 82 + "\n", flush=True)

    return {
        "status": "success",
        "trip_id": trip_obj.trip_id,
        "partition": trip_obj.partition,
        "model_used": meta.get("selected_model_name", "Bayesian-Tuned LightGBM"),
        "metrics": {k: v for k, v in metrics.items() if k not in {"output_route_file", "map_html_path"}},
        "reconstructed_route_arc_ids": pred_result["reconstructed_route"],
        "observation_predicted_arc_ids": pred_result["hmm_pred_arcs"],
        "folium_map_html": map_html_str,
    }
