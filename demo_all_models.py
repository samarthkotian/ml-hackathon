"""
Quick Phase-1 Evaluator Demo Script — Team SI No 107.
Displays the complete benchmark across all 7 models (6 algorithm families),
Bayesian Hyperparameter Optimisation (Optuna TPE), and runs a live end-to-end
route reconstruction on Student-Test Trip 00000007 in < 1 second.

Usage:
    python demo_all_models.py
"""
import json
import os
import pandas as pd

from src.config import MODELS_DIR, RESULTS_OUT_DIR, STUDENT_TEST_DIR, TEAM_INFO
from src.data_loader import load_trip_from_dir
from predict import load_inference_artifacts, predict_single_trip


def main() -> None:
    print("Saved benchmark tables are historical unless marked wta-lcs-v2; the live prediction below uses the corrected protocol. See docs/EVALUATION_CORRECTIONS.md.")
    print("=" * 95)
    print(" PATHSNAP PHASE 1 EVALUATION DEMO — TEAM SI No 107 (PES UNIVERSITY)")
    for srn, name in TEAM_INFO["members"]:
        print(f"   • {srn} — {name}")
    print("=" * 95)

    # 1. Show all 7 models benchmark table
    df_models = pd.read_csv(os.path.join(RESULTS_OUT_DIR, "model_comparison_results.csv"))
    print("\n[1] ALL 7 TRAINED MODELS ACROSS 6 ALGORITHM FAMILIES (80 Train Trips 5-Fold GroupCV + 10 Test Trips):")
    print("-" * 95)
    display_df = pd.DataFrame({
        "Model": df_models["model_name"],
        "Algorithm Family": df_models["algorithm_family"],
        "CV Macro F1": df_models["cv_macro_f1_mean"].map(lambda x: f"{x:.4f}") + " ± " + df_models["cv_macro_f1_std"].map(lambda x: f"{x:.4f}"),
        "Test Top-1 Acc": (df_models["student_test_top1_accuracy"] * 100).map(lambda x: f"{x:.2f}%"),
        "Test Macro F1": df_models["student_test_macro_f1"].map(lambda x: f"{x:.4f}"),
        "Log Loss": df_models["student_test_log_loss"].map(lambda x: f"{x:.4f}"),
        "Continuity": (df_models["route_continuity"] * 100).map(lambda x: f"{x:.2f}%"),
        "Latency": df_models["inference_ms_per_trip"].map(lambda x: f"{x:.1f} ms"),
    })
    print(display_df.to_string(index=False))

    # 2. Show Bayesian Optimisation Improvement
    df_bo = pd.read_csv(os.path.join(RESULTS_OUT_DIR, "bayesian_untuned_vs_tuned.csv"))
    print("\n[2] BAYESIAN HYPERPARAMETER OPTIMISATION (OPTUNA TPE — 16 TRIALS, 80 FITS):")
    print("-" * 95)
    bo_disp = pd.DataFrame({
        "Model Version": df_bo["version"],
        "5-Fold CV Macro F1": df_bo["cv_macro_f1"].map(lambda x: f"{x:.4f}"),
        "Student-Test Top-1 Acc": (df_bo["student_test_top1_accuracy"] * 100).map(lambda x: f"{x:.2f}%"),
        "Student-Test Macro F1": df_bo["student_test_macro_f1"].map(lambda x: f"{x:.4f}"),
        "Log Loss": df_bo["student_test_log_loss"].map(lambda x: f"{x:.4f}"),
        "Route Continuity": (df_bo["route_continuity"] * 100).map(lambda x: f"{x:.2f}%"),
    })
    print(bo_disp.to_string(index=False))

    # 3. Show HMM + Viterbi Route Reconstruction Improvement
    with open(os.path.join(RESULTS_OUT_DIR, "hmm_summary.json"), "r", encoding="utf-8") as f:
        hmm_sum = json.load(f)
    print("\n[3] SEQUENTIAL ROUTE RECONSTRUCTION ABLATION (CLASSIFIER-ONLY vs CLASSIFIER + HMM/VITERBI):")
    print("-" * 95)
    print(f"  • Classifier-Only (Pointwise): Top-1 Acc = {hmm_sum['overall_clf_top1_accuracy']*100:.2f}% | Macro F1 = {hmm_sum['overall_clf_macro_f1']:.4f} | Route Continuity = {hmm_sum['mean_clf_route_continuity']*100:.2f}% | Seq Route F1 = {hmm_sum['mean_clf_seq_f1']:.4f}")
    print(f"  • Classifier + HMM/Viterbi:    Top-1 Acc = {hmm_sum['overall_hmm_top1_accuracy']*100:.2f}% | Macro F1 = {hmm_sum['overall_hmm_macro_f1']:.4f} | Route Continuity = {hmm_sum['mean_hmm_route_continuity']*100:.2f}% | Seq Route F1 = {hmm_sum['mean_hmm_seq_f1']:.4f}")

    # 4. Run live inference on Sample Student-Test Trip 00000007
    print("\n[4] LIVE MODEL & VITERBI EXECUTION CHECK ON TRIP 00000007:")
    print("-" * 95)
    model, hmm_cfg, meta = load_inference_artifacts()
    trip_dir = os.path.join(STUDENT_TEST_DIR, "00000007")
    trip = load_trip_from_dir(trip_dir, "00000007", partition="student_test")
    res = predict_single_trip(trip, model, hmm_cfg, save_outputs=True, generate_map=False)
    m = res["metrics"]
    print(f"  ✓ Loaded Artifact: {meta.get('selected_model_name')} (models/pathsnap_best_model.joblib)")
    print(f"  ✓ Input Trajectory: {m['n_observations']} GPS points across {m['n_arcs']} road segments")
    print(f"  ✓ Live Top-1 Accuracy: {m['hmm_top1_accuracy']*100:.2f}% (Classifier-Only: {m['clf_top1_accuracy']*100:.2f}%)")
    print(f"  ✓ Live Candidate Macro F1: {m['macro_f1']:.4f} | Sequence Route F1: {m['route_seq_f1']:.4f}")
    print(f"  ✓ Live Route Continuity: {m['route_continuity']*100:.2f}% | Latency: {m['inference_ms']:.1f} ms")
    print("=" * 95)


if __name__ == "__main__":
    main()
