"""
One-command inference and route reconstruction script for PathSnap (Team SI No 107).

Loads the saved Bayesian-tuned classifier and HMM/Viterbi configuration from `models/`,
processes trips in `student_test` or `evaluator_test` (or any specified `--trip_dir`),
outputs reconstructed `.route` files in the exact single-column integer format required,
validates output format and road-network arc_id ranges, and prints server-side metrics.

Usage:
    python predict.py
    python predict.py --partition evaluator_test
    python predict.py --trip_dir "Data/Student folder/student_test/00000000"
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import (
    EVALUATOR_TEST_DIR,
    MAPS_OUT_DIR,
    MODELS_DIR,
    PREDICTIONS_OUT_DIR,
    STUDENT_TEST_DIR,
    TOP_K_CANDIDATES,
)
from src.data_loader import TripData, load_partition, load_trip_from_dir
from src.features import extract_features_for_trip
from src.hmm_viterbi import (
    HMMConfig,
    compute_route_continuity,
    compute_sequence_route_metrics,
    viterbi_decode_trip,
)
from src.visualization import generate_folium_map_for_trip


def load_inference_artifacts() -> tuple[Any, HMMConfig, Dict[str, Any]]:
    """Load trained classifier, HMM configuration, and metadata from `models/`."""
    model_path = os.path.join(MODELS_DIR, "pathsnap_best_model.joblib")
    hmm_cfg_path = os.path.join(MODELS_DIR, "hmm_config.json")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python train.py` first."
        )

    model = joblib.load(model_path)
    if os.path.exists(hmm_cfg_path):
        with open(hmm_cfg_path, "r", encoding="utf-8") as f:
            hmm_cfg = HMMConfig(**json.load(f))
    else:
        hmm_cfg = HMMConfig()

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, hmm_cfg, metadata


def predict_single_trip(
    trip: TripData,
    model: Any,
    hmm_cfg: HMMConfig,
    save_outputs: bool = True,
    generate_map: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete PathSnap inference pipeline on a single TripData instance.
    Note: Ground-truth `.route` is NEVER used during candidate generation, feature
    extraction, classification, or Viterbi decoding. It is used strictly post-hoc
    for server-side evaluation metrics if present.
    """
    t0 = time.perf_counter()

    # 1. Extract spatial/kinematic/topological features & candidates
    X_trip, _, candidates = extract_features_for_trip(trip)
    n_obs, k = candidates.cand_arcs.shape

    # 2. Predict candidate emission probabilities using trained classifier
    clf_probs_flat = model.predict_proba(X_trip)[:, 1]
    clf_probs_2d = clf_probs_flat.reshape(n_obs, k)

    # Pointwise classifier-only prediction
    clf_best_k = np.argmax(clf_probs_2d, axis=1)
    clf_pred_arcs = candidates.cand_arcs[np.arange(n_obs), clf_best_k]

    # 3. Sequential Route Reconstruction via HMM & Log-Domain Viterbi
    hmm_pred_arcs, reconstructed_route, hmm_stats = viterbi_decode_trip(
        trip, candidates, clf_probs_2d, config=hmm_cfg
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Validate predictions against road network (.arcs valid 0-based index range)
    n_arcs = len(trip.arcs)
    assert len(hmm_pred_arcs) == n_obs, "Observation prediction count mismatch!"
    assert np.all((hmm_pred_arcs >= 0) & (hmm_pred_arcs < n_arcs)), "Invalid arc_id predicted!"
    assert len(reconstructed_route) >= 1, "Reconstructed route cannot be empty!"
    assert all(0 <= aid < n_arcs for aid in reconstructed_route), "Route arc_id out of range!"

    # 5. Save formatted output files
    out_route_file = os.path.join(PREDICTIONS_OUT_DIR, f"{trip.trip_id}.route")
    out_obs_file = os.path.join(PREDICTIONS_OUT_DIR, f"{trip.trip_id}_obs_arcs.txt")
    if save_outputs:
        os.makedirs(PREDICTIONS_OUT_DIR, exist_ok=True)
        np.savetxt(out_route_file, np.array(reconstructed_route, dtype=np.int32), fmt="%d")
        np.savetxt(out_obs_file, hmm_pred_arcs, fmt="%d")

    map_html_path = None
    if generate_map:
        map_html_path = os.path.join(MAPS_OUT_DIR, f"predicted_trip_{trip.trip_id}.html")
        generate_folium_map_for_trip(
            trip,
            clf_only_arcs=clf_pred_arcs,
            hmm_pred_arcs=hmm_pred_arcs,
            hmm_reconstructed_route=reconstructed_route,
            out_html_path=map_html_path,
        )

    # 6. Server-side evaluation against internal reference data (if available)
    eval_metrics: Dict[str, Any] = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_observations": int(n_obs),
        "n_nodes": int(len(trip.nodes)),
        "n_arcs": int(n_arcs),
        "n_reconstructed_route_arcs": int(len(reconstructed_route)),
        "route_continuity": float(hmm_stats["route_continuity"]),
        "strict_1hop_continuity": float(hmm_stats["strict_1hop_continuity"]),
        "clf_only_continuity": float(compute_route_continuity(trip, clf_pred_arcs)["connected_continuity"]),
        "inference_ms": float(inference_ms),
        "has_ground_truth": bool(trip.obs_ref_arcs is not None and len(trip.obs_ref_arcs) == n_obs),
        "output_route_file": out_route_file,
        "map_html_path": map_html_path,
    }

    if eval_metrics["has_ground_truth"]:
        clf_top1 = float(np.mean(clf_pred_arcs == trip.obs_ref_arcs))
        hmm_top1 = float(np.mean(hmm_pred_arcs == trip.obs_ref_arcs))
        hmm_wta = (candidates.cand_arcs == hmm_pred_arcs[:, None]).astype(np.int32)
        macro_f1 = float(
            f1_score(candidates.labels.ravel(), hmm_wta.ravel(), average="macro", zero_division=0)
        )
        seq_metrics = compute_sequence_route_metrics(reconstructed_route, trip.route)

        eval_metrics.update(
            {
                "clf_top1_accuracy": clf_top1,
                "hmm_top1_accuracy": hmm_top1,
                "top1_improvement_pct": (hmm_top1 - clf_top1) * 100.0,
                "macro_f1": macro_f1,
                "candidate_miss_rate": float(candidates.candidate_miss_rate or 0.0),
                "route_seq_f1": float(seq_metrics["route_seq_f1"]),
                "route_jaccard": float(seq_metrics["route_jaccard"]),
            }
        )

    return {
        "metrics": eval_metrics,
        "clf_pred_arcs": clf_pred_arcs.tolist(),
        "hmm_pred_arcs": hmm_pred_arcs.tolist(),
        "reconstructed_route": reconstructed_route,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Inference & Route Reconstruction CLI")
    parser.add_argument(
        "--partition",
        type=str,
        default="auto",
        choices=["auto", "student_test", "evaluator_test"],
        help="Dataset partition to evaluate ('auto' runs evaluator_test if available, plus student_test)",
    )
    parser.add_argument(
        "--trip_dir",
        type=str,
        default=None,
        help="Optional path to a specific trip folder containing .track, .nodes, .arcs",
    )
    parser.add_argument(
        "--generate_maps",
        action="store_true",
        help="Generate interactive Folium HTML maps for predicted trips",
    )
    args = parser.parse_args()

    model, hmm_cfg, meta = load_inference_artifacts()
    print("=" * 80)
    print("PATHSNAP: ROUTE RECONSTRUCTION INFERENCE ENGINE")
    print(f"Loaded Model: {meta.get('selected_model_name', 'Bayesian-Tuned LightGBM')}")
    print("=" * 80)

    trips_to_run: List[TripData] = []
    if args.trip_dir is not None:
        tid = os.path.basename(os.path.normpath(args.trip_dir))
        trips_to_run.append(load_trip_from_dir(args.trip_dir, tid, partition="custom"))
    else:
        if args.partition in ("auto", "evaluator_test") and os.path.exists(EVALUATOR_TEST_DIR):
            eval_trips = load_partition("evaluator_test", EVALUATOR_TEST_DIR)
            if eval_trips:
                print(f"Found {len(eval_trips)} trips in evaluator_test partition.")
                trips_to_run.extend(eval_trips)
        if args.partition in ("auto", "student_test") and os.path.exists(STUDENT_TEST_DIR):
            stu_trips = load_partition("student_test", STUDENT_TEST_DIR)
            print(f"Running inference on {len(stu_trips)} trips in student_test partition.")
            trips_to_run.extend(stu_trips)

    if not trips_to_run:
        print("No trips found to run inference on.")
        return

    rows = []
    for trip in trips_to_run:
        res = predict_single_trip(
            trip, model, hmm_cfg, save_outputs=True, generate_map=args.generate_maps
        )
        m = res["metrics"]
        rows.append(m)
        acc_str = (
            f"Top-1 Acc: {m['hmm_top1_accuracy']*100:.2f}% (Clf-Only: {m['clf_top1_accuracy']*100:.2f}%) | Macro F1: {m['macro_f1']:.4f}"
            if m["has_ground_truth"]
            else "Top-1 Acc: N/A (Unlabelled)"
        )
        print(
            f"[Trip {m['trip_id']}] Obs={m['n_observations']:4d} | PredRouteLen={m['n_reconstructed_route_arcs']:3d} | "
            f"{acc_str} | Continuity={m['route_continuity']*100:.2f}% | Latency={m['inference_ms']:.1f}ms -> {m['output_route_file']}"
        )

    df_out = pd.DataFrame(rows)
    summary_csv = os.path.join(PREDICTIONS_OUT_DIR, "inference_validation_summary.csv")
    df_out.to_csv(summary_csv, index=False)
    print("-" * 80)
    if df_out["has_ground_truth"].any():
        labelled = df_out[df_out["has_ground_truth"]]
        print(
            f"AGGREGATE ({len(labelled)} trips) -> "
            f"Mean Top-1 Accuracy: {labelled['hmm_top1_accuracy'].mean()*100:.2f}% | "
            f"Mean Macro F1: {labelled['macro_f1'].mean():.4f} | "
            f"Mean Route Continuity: {labelled['route_continuity'].mean()*100:.2f}% | "
            f"Mean Latency: {labelled['inference_ms'].mean():.1f} ms/trip"
        )
    print(f"Saved validation summary to: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
