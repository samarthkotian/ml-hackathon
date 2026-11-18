"""
Phase 2 Experimentation, Ensembling, and Verification Pipeline for PathSnap (Team SI No 107).

Logs every Phase 2 improvement experiment in a structured comparison table
(`outputs/results/phase1_vs_phase2_comparison.csv`) without ever training or tuning
on `evaluator_test/`. Saves the winning Phase 2 model to `models/pathsnap_best_model.joblib`,
updates `models/hmm_config.json`, and generates all predictions and Folium maps for
both `evaluator_test` (10 trips) and `student_test` (10 trips).
"""
from dataclasses import asdict
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    EVALUATOR_TEST_DIR,
    FIGURES_OUT_DIR,
    MAPS_OUT_DIR,
    MODELS_DIR,
    PREDICTIONS_OUT_DIR,
    RANDOM_SEED,
    RESULTS_OUT_DIR,
    STUDENT_TEST_DIR,
    TOP_K_CANDIDATES,
    TRAIN_DIR,
    set_global_seed,
)
from src.data_loader import TripData, load_partition
from src.features import build_partition_dataset
from src.hmm_viterbi import HMMConfig
from src.models import subsample_trip_preserving_observations
from src.visualization import (
    generate_folium_map_for_trip,
    generate_static_map_figure_for_trip,
)
from predict import predict_single_trip


def evaluate_model_on_partition(
    trips: List[TripData],
    model: Any,
    hmm_cfg: HMMConfig,
    save_outputs: bool = False,
    generate_maps_for: tuple = (),
    pred_subdir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    total_obs = 0
    total_clf_hits = 0
    total_hmm_hits = 0

    for trip in trips:
        res = predict_single_trip(
            trip, model, hmm_cfg, save_outputs=save_outputs, generate_map=False
        )
        m = res["metrics"]
        rows.append(m)

        n_obs = m["n_observations"]
        total_obs += n_obs
        if m["has_ground_truth"]:
            total_clf_hits += int(round(m["clf_top1_accuracy"] * n_obs))
            total_hmm_hits += int(round(m["hmm_top1_accuracy"] * n_obs))

        if pred_subdir is not None:
            sub_p = os.path.join(PREDICTIONS_OUT_DIR, pred_subdir)
            os.makedirs(sub_p, exist_ok=True)
            np.savetxt(
                os.path.join(sub_p, f"{trip.trip_id}.route"),
                np.array(res["reconstructed_route"], dtype=np.int32),
                fmt="%d",
            )
            np.savetxt(
                os.path.join(sub_p, f"{trip.trip_id}_obs_arcs.txt"),
                np.array(res["hmm_pred_arcs"], dtype=np.int32),
                fmt="%d",
            )

        if trip.trip_id in generate_maps_for:
            html_p = os.path.join(MAPS_OUT_DIR, f"folium_trip_{trip.trip_id}.html")
            generate_folium_map_for_trip(
                trip,
                clf_only_arcs=np.array(res["clf_pred_arcs"], dtype=int),
                hmm_pred_arcs=np.array(res["hmm_pred_arcs"], dtype=int),
                hmm_reconstructed_route=res["reconstructed_route"],
                out_html_path=html_p,
            )
            png_p = os.path.join(MAPS_OUT_DIR, f"map_comparison_{trip.trip_id}.png")
            generate_static_map_figure_for_trip(
                trip,
                clf_only_arcs=np.array(res["clf_pred_arcs"], dtype=int),
                hmm_pred_arcs=np.array(res["hmm_pred_arcs"], dtype=int),
                hmm_reconstructed_route=res["reconstructed_route"],
                out_png_path=png_p,
            )

    df = pd.DataFrame(rows)
    agg = {
        "mean_clf_top1_accuracy": float(df["clf_top1_accuracy"].mean()),
        "mean_hmm_top1_accuracy": float(df["hmm_top1_accuracy"].mean()),
        "overall_clf_top1_accuracy": float(total_clf_hits / max(1, total_obs)),
        "overall_hmm_top1_accuracy": float(total_hmm_hits / max(1, total_obs)),
        "mean_macro_f1": float(df["macro_f1"].mean()),
        "mean_route_continuity": float(df["route_continuity"].mean()),
        "mean_clf_continuity": float(df["clf_only_continuity"].mean()),
        "mean_route_seq_f1": float(df["route_seq_f1"].mean()),
        "mean_candidate_miss_rate": float(df["candidate_miss_rate"].mean()),
        "mean_inference_ms": float(df["inference_ms"].mean()),
    }
    return df, agg


def main() -> None:
    set_global_seed(RANDOM_SEED)
    print("=" * 88)
    print(" PATHSNAP PHASE 2 — LEAKAGE-FREE PIPELINE IMPROVEMENTS & EVALUATOR VERIFICATION")
    print("=" * 88)

    # Load partitions (evaluator_test is strictly read-only evaluation data!)
    train_trips = load_partition("train", TRAIN_DIR)
    student_trips = load_partition("student_test", STUDENT_TEST_DIR)
    evaluator_trips = load_partition("evaluator_test", EVALUATOR_TEST_DIR)
    print(
        f"Loaded {len(train_trips)} train trips, {len(student_trips)} student_test trips, "
        f"and {len(evaluator_trips)} evaluator_test trips (from {EVALUATOR_TEST_DIR})."
    )

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_lgbm_params = meta["best_hyperparameters"]

    # Record Experiment 1: Phase 1 Baseline (measured before Phase 2 spatial-chord & U-turn upgrades)
    exp_rows = [
        {
            "experiment_id": "Exp 1 (Phase 1 Baseline)",
            "description": "Phase 1 Bayesian-Tuned LightGBM + 5-Step Heading + Phase 1 HMM (No Twin Tie-Break)",
            "student_test_mean_top1": 0.921996,
            "student_test_obs_top1": 0.911326,
            "student_test_macro_f1": 0.956803,
            "student_test_continuity": 0.996588,
            "student_test_seq_f1": 0.939429,
            "evaluator_test_mean_top1": 0.927491,
            "evaluator_test_obs_top1": 0.896818,
            "evaluator_test_macro_f1": 0.960651,
            "evaluator_test_continuity": 0.995749,
            "evaluator_test_seq_f1": 0.908192,
            "evaluator_test_clf_only_top1": 0.905627,
            "evaluator_test_latency_ms": 448.6,
        }
    ]

    # Experiment 2: Existing Phase 1 Classifier + Phase 2 Spatial-Chord Tie-Break & Directional HMM
    print("\n[Experiment 2] Evaluating Phase 1 Classifier + Phase 2 Spatial-Chord & Directional HMM...")
    p1_model = joblib.load(os.path.join(MODELS_DIR, "pathsnap_best_model.joblib"))
    p2_hmm_cfg = HMMConfig()
    _, stu_agg_e2 = evaluate_model_on_partition(student_trips, p1_model, p2_hmm_cfg)
    _, eval_agg_e2 = evaluate_model_on_partition(evaluator_trips, p1_model, p2_hmm_cfg)

    exp_rows.append(
        {
            "experiment_id": "Exp 2 (+Spatial-Chord & Dir-HMM)",
            "description": "Phase 1 Classifier + 18m Spatial-Chord Twin Tie-Break + U-Turn-Aware Directional HMM",
            "student_test_mean_top1": stu_agg_e2["mean_hmm_top1_accuracy"],
            "student_test_obs_top1": stu_agg_e2["overall_hmm_top1_accuracy"],
            "student_test_macro_f1": stu_agg_e2["mean_macro_f1"],
            "student_test_continuity": stu_agg_e2["mean_route_continuity"],
            "student_test_seq_f1": stu_agg_e2["mean_route_seq_f1"],
            "evaluator_test_mean_top1": eval_agg_e2["mean_hmm_top1_accuracy"],
            "evaluator_test_obs_top1": eval_agg_e2["overall_hmm_top1_accuracy"],
            "evaluator_test_macro_f1": eval_agg_e2["mean_macro_f1"],
            "evaluator_test_continuity": eval_agg_e2["mean_route_continuity"],
            "evaluator_test_seq_f1": eval_agg_e2["mean_route_seq_f1"],
            "evaluator_test_clf_only_top1": eval_agg_e2["mean_clf_top1_accuracy"],
            "evaluator_test_latency_ms": eval_agg_e2["mean_inference_ms"],
        }
    )
    print(
        f"  -> Student-Test Mean Top-1: {stu_agg_e2['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {stu_agg_e2['overall_hmm_top1_accuracy']*100:.2f}%) | "
        f"Evaluator-Test Mean Top-1: {eval_agg_e2['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {eval_agg_e2['overall_hmm_top1_accuracy']*100:.2f}%)"
    )

    # Build Phase 2 refined feature matrix on the 80 training trips (zero evaluator_test data used!)
    print("\n[Experiment 3] Retraining Bayesian-Tuned LightGBM on Phase 2 Spatial-Chord Features (80 Train Trips)...")
    X_train, y_train, groups_train, _ = build_partition_dataset(train_trips)
    X_fit, y_fit, _ = subsample_trip_preserving_observations(
        X_train, y_train, groups_train, TOP_K_CANDIDATES, 650000, seed=RANDOM_SEED
    )

    lgbm_p2 = LGBMClassifier(
        **best_lgbm_params,
        subsample_freq=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_p2.fit(X_fit, y_fit)

    _, stu_agg_e3 = evaluate_model_on_partition(student_trips, lgbm_p2, p2_hmm_cfg)
    _, eval_agg_e3 = evaluate_model_on_partition(evaluator_trips, lgbm_p2, p2_hmm_cfg)

    exp_rows.append(
        {
            "experiment_id": "Exp 3 (Retrained P2 LightGBM)",
            "description": "Retrained Bayesian-Tuned LightGBM on Phase 2 Spatial-Chord Features + Directional HMM",
            "student_test_mean_top1": stu_agg_e3["mean_hmm_top1_accuracy"],
            "student_test_obs_top1": stu_agg_e3["overall_hmm_top1_accuracy"],
            "student_test_macro_f1": stu_agg_e3["mean_macro_f1"],
            "student_test_continuity": stu_agg_e3["mean_route_continuity"],
            "student_test_seq_f1": stu_agg_e3["mean_route_seq_f1"],
            "evaluator_test_mean_top1": eval_agg_e3["mean_hmm_top1_accuracy"],
            "evaluator_test_obs_top1": eval_agg_e3["overall_hmm_top1_accuracy"],
            "evaluator_test_macro_f1": eval_agg_e3["mean_macro_f1"],
            "evaluator_test_continuity": eval_agg_e3["mean_route_continuity"],
            "evaluator_test_seq_f1": eval_agg_e3["mean_route_seq_f1"],
            "evaluator_test_clf_only_top1": eval_agg_e3["mean_clf_top1_accuracy"],
            "evaluator_test_latency_ms": eval_agg_e3["mean_inference_ms"],
        }
    )
    print(
        f"  -> Student-Test Mean Top-1: {stu_agg_e3['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {stu_agg_e3['overall_hmm_top1_accuracy']*100:.2f}%) | "
        f"Evaluator-Test Mean Top-1: {eval_agg_e3['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {eval_agg_e3['overall_hmm_top1_accuracy']*100:.2f}%)"
    )

    # Experiment 4: Soft-Voting GBDT Ensemble (Bayesian-Tuned LightGBM + Regularised XGBoost)
    print("\n[Experiment 4] Training Soft-Voting GBDT Ensemble (Tuned LightGBM + XGBoost) on 80 Train Trips...")
    xgb_p2 = XGBClassifier(
        n_estimators=240,
        learning_rate=0.06,
        max_depth=6,
        colsample_bytree=0.85,
        subsample=0.85,
        min_child_weight=8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    ensemble_p2 = VotingClassifier(
        estimators=[
            (
                "lgbm",
                LGBMClassifier(
                    **best_lgbm_params,
                    subsample_freq=1,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
            ("xgb", xgb_p2),
        ],
        voting="soft",
        weights=[0.60, 0.40],
    )
    ensemble_p2.fit(X_fit, y_fit)

    _, stu_agg_e4 = evaluate_model_on_partition(student_trips, ensemble_p2, p2_hmm_cfg)
    _, eval_agg_e4 = evaluate_model_on_partition(evaluator_trips, ensemble_p2, p2_hmm_cfg)

    exp_rows.append(
        {
            "experiment_id": "Exp 4 (P2 LightGBM+XGB Ensemble)",
            "description": "Soft-Voting GBDT Ensemble (0.6 Tuned LightGBM + 0.4 XGBoost) + Directional HMM",
            "student_test_mean_top1": stu_agg_e4["mean_hmm_top1_accuracy"],
            "student_test_obs_top1": stu_agg_e4["overall_hmm_top1_accuracy"],
            "student_test_macro_f1": stu_agg_e4["mean_macro_f1"],
            "student_test_continuity": stu_agg_e4["mean_route_continuity"],
            "student_test_seq_f1": stu_agg_e4["mean_route_seq_f1"],
            "evaluator_test_mean_top1": eval_agg_e4["mean_hmm_top1_accuracy"],
            "evaluator_test_obs_top1": eval_agg_e4["overall_hmm_top1_accuracy"],
            "evaluator_test_macro_f1": eval_agg_e4["mean_macro_f1"],
            "evaluator_test_continuity": eval_agg_e4["mean_route_continuity"],
            "evaluator_test_seq_f1": eval_agg_e4["mean_route_seq_f1"],
            "evaluator_test_clf_only_top1": eval_agg_e4["mean_clf_top1_accuracy"],
            "evaluator_test_latency_ms": eval_agg_e4["mean_inference_ms"],
        }
    )
    print(
        f"  -> Student-Test Mean Top-1: {stu_agg_e4['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {stu_agg_e4['overall_hmm_top1_accuracy']*100:.2f}%) | "
        f"Evaluator-Test Mean Top-1: {eval_agg_e4['mean_hmm_top1_accuracy']*100:.2f}% (Obs: {eval_agg_e4['overall_hmm_top1_accuracy']*100:.2f}%)"
    )

    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(os.path.join(RESULTS_OUT_DIR, "phase1_vs_phase2_comparison.csv"), index=False)
    print("\n=== PHASE 1 vs PHASE 2 EXPERIMENT PROGRESSION ===")
    print(exp_df.to_string(index=False))

    # Select best model strictly by student_test development score (Exp 3 vs Exp 4)
    if stu_agg_e4["overall_hmm_top1_accuracy"] >= stu_agg_e3["overall_hmm_top1_accuracy"]:
        winning_model = ensemble_p2
        winning_name = "Phase-2 GBDT Ensemble (Bayesian-Tuned LightGBM + XGBoost)"
    else:
        winning_model = lgbm_p2
        winning_name = "Phase-2 Bayesian-Tuned LightGBM (Spatial-Chord & Directional HMM)"

    # Also keep lgbm_p2 saved for SHAP TreeExplainer compatibility if needed
    joblib.dump(lgbm_p2, os.path.join(MODELS_DIR, "pathsnap_lgbm_phase2.joblib"))
    joblib.dump(winning_model, os.path.join(MODELS_DIR, "pathsnap_best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "hmm_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(p2_hmm_cfg), f, indent=2)

    # Generate final predictions and maps on both student_test and evaluator_test using winning_model
    print(f"\n[Finalizing Outputs] Running {winning_name} on student_test and evaluator_test...")
    stu_final_df, stu_final_agg = evaluate_model_on_partition(
        student_trips,
        winning_model,
        p2_hmm_cfg,
        save_outputs=True,
        generate_maps_for=("00000000", "00000007", "00000052", "00000077"),
        pred_subdir="student_test",
    )
    eval_final_df, eval_final_agg = evaluate_model_on_partition(
        evaluator_trips,
        winning_model,
        p2_hmm_cfg,
        save_outputs=True,
        generate_maps_for=("00000015", "00000049", "00000087"),
        pred_subdir="evaluator_test",
    )

    stu_final_df.to_csv(os.path.join(RESULTS_OUT_DIR, "student_test_phase2_results.csv"), index=False)
    eval_final_df.to_csv(os.path.join(RESULTS_OUT_DIR, "evaluator_test_results.csv"), index=False)

    meta["selected_model_name"] = winning_name
    meta["phase2_student_test_summary"] = stu_final_agg
    meta["phase2_evaluator_test_summary"] = eval_final_agg
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== FINAL EVALUATOR_TEST PER-TRIP RESULTS (10 TRIPS) ===")
    cols_show = [
        "trip_id",
        "n_observations",
        "n_reconstructed_route_arcs",
        "candidate_miss_rate",
        "clf_top1_accuracy",
        "hmm_top1_accuracy",
        "top1_improvement_pct",
        "macro_f1",
        "route_continuity",
        "route_seq_f1",
        "inference_ms",
    ]
    print(eval_final_df[cols_show].to_string(index=False))
    print("-" * 88)
    print(
        f"EVALUATOR_TEST FINAL AGGREGATE -> "
        f"Mean Trip Top-1: {eval_final_agg['mean_hmm_top1_accuracy']*100:.2f}% | "
        f"Overall Obs Top-1: {eval_final_agg['overall_hmm_top1_accuracy']*100:.2f}% | "
        f"Mean Macro F1: {eval_final_agg['mean_macro_f1']:.4f} | "
        f"Mean Route Continuity: {eval_final_agg['mean_route_continuity']*100:.2f}% | "
        f"Mean Seq Route F1: {eval_final_agg['mean_route_seq_f1']:.4f}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
