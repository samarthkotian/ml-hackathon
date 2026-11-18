"""
One-command end-to-end training, cross-validation, Bayesian hyperparameter optimisation,
HMM/Viterbi route reconstruction evaluation, and artifact generation for PathSnap (Team SI No 107).

Usage:
    python train.py
"""
import json
import os
import time
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import (
    EDA_OUT_DIR,
    FIGURES_OUT_DIR,
    MAPS_OUT_DIR,
    MODELS_DIR,
    PREDICTIONS_OUT_DIR,
    RANDOM_SEED,
    RESULTS_OUT_DIR,
    TOP_K_CANDIDATES,
    TEAM_INFO,
    set_global_seed,
)
from src.data_loader import load_partition
from src.eda import run_eda_and_save_artifacts
from src.features import FEATURE_NAMES, build_partition_dataset
from src.hmm_viterbi import (
    HMMConfig,
    compute_route_continuity,
    compute_sequence_route_metrics,
    viterbi_decode_trip,
)
from src.models import run_model_experimentation
from src.tuning import run_bayesian_optimisation
from src.visualization import (
    generate_folium_map_for_trip,
    generate_static_map_figure_for_trip,
    plot_bayesian_optimisation,
    plot_confusion_matrices,
    plot_epoch_loss_curves,
    plot_feature_importance_and_shap,
    plot_hmm_ablation,
    plot_model_comparisons,
)


def main() -> None:
    t_start = time.perf_counter()
    set_global_seed(RANDOM_SEED)
    print("=" * 80)
    print("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION — TRAINING PIPELINE")
    print(f"Team: {TEAM_INFO['team_id']} | Random Seed: {RANDOM_SEED}")
    print("=" * 80)

    # 1. Load Raw Datasets
    print("\n[Step 1/7] Loading Training (80 trips) and Student-Testing (10 trips) partitions...")
    train_trips = load_partition("train")
    test_trips = load_partition("student_test")
    print(f"  -> Loaded {len(train_trips)} training trips and {len(test_trips)} student-testing trips.")

    # 2. Run Exploratory Data Analysis
    print("\n[Step 2/7] Running Exploratory Data Analysis & Data Quality Audit...")
    inv_df, part_df, eda_stats = run_eda_and_save_artifacts(train_trips, test_trips, out_dir=EDA_OUT_DIR)
    print("  -> Partition Summary:")
    print(part_df.to_string(index=False))

    # 3. Candidate Generation & Feature Engineering
    print(f"\n[Step 3/7] Generating Top-{TOP_K_CANDIDATES} Spatial Candidates & Extracting {len(FEATURE_NAMES)} Features...")
    t0_fe = time.perf_counter()
    X_train, y_train, groups_train, train_cands = build_partition_dataset(train_trips)
    X_test, y_test, groups_test, test_cands = build_partition_dataset(test_trips)
    fe_time = time.perf_counter() - t0_fe

    train_miss_rates = [c.candidate_miss_rate for c in train_cands.values() if c.candidate_miss_rate is not None]
    test_miss_rates = [c.candidate_miss_rate for c in test_cands.values() if c.candidate_miss_rate is not None]
    mean_train_miss = float(np.mean(train_miss_rates))
    mean_test_miss = float(np.mean(test_miss_rates))

    print(f"  -> Feature matrix X_train: {X_train.shape} | Positive ratio: {np.mean(y_train)*100:.2f}%")
    print(f"  -> Feature matrix X_test:  {X_test.shape}  | Positive ratio: {np.mean(y_test)*100:.2f}%")
    print(f"  -> Candidate Miss Rate (Train): {mean_train_miss*100:.2f}% | (Student-Test): {mean_test_miss*100:.2f}%")
    print(f"  -> Feature extraction completed in {fe_time:.2f}s.")

    # 4. Multi-Family Model Experimentation (7 Models across 6 Algorithm Families)
    print("\n[Step 4/7] Running Multi-Family Model Experimentation (5-Fold GroupKFold CV)...")
    df_models, trained_models, diagnostics = run_model_experimentation(
        train_trips=train_trips,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        train_cands=train_cands,
        test_trips=test_trips,
        X_test=X_test,
        y_test=y_test,
        test_cands=test_cands,
        k_candidates=TOP_K_CANDIDATES,
    )
    df_models["candidate_miss_rate"] = mean_test_miss
    df_models.to_csv(os.path.join(RESULTS_OUT_DIR, "model_comparison_results.csv"), index=False)

    # Save serializable diagnostics (fold scores, confusion matrices, epoch loss histories)
    serializable_diag = {
        name: {
            "fold_f1s": d["fold_f1s"],
            "fold_top1s": d["fold_top1s"],
            "fold_lls": d["fold_lls"],
            "confusion_matrix": d["confusion_matrix"],
            "epoch_history": d["epoch_history"],
        }
        for name, d in diagnostics.items()
    }
    with open(os.path.join(RESULTS_OUT_DIR, "model_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_diag, f, indent=2)

    # 5. Bayesian Hyperparameter Optimisation on Selected Model (LightGBM)
    print("\n[Step 5/7] Performing Bayesian Hyperparameter Optimisation (Optuna TPE)...")
    lgbm_untuned_row = df_models.loc[df_models["model_name"] == "LightGBM"].iloc[0].to_dict()
    best_model, trials_df, tuned_comp_df, tuning_summary = run_bayesian_optimisation(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        test_trips=test_trips,
        X_test=X_test,
        y_test=y_test,
        test_cands=test_cands,
        untuned_row=lgbm_untuned_row,
        n_trials=16,
        k_candidates=TOP_K_CANDIDATES,
    )
    trials_df.to_csv(os.path.join(RESULTS_OUT_DIR, "bayesian_trials.csv"), index=False)
    tuned_comp_df.to_csv(os.path.join(RESULTS_OUT_DIR, "bayesian_untuned_vs_tuned.csv"), index=False)

    # 6. HMM and Viterbi Sequential Route Reconstruction & Ablation Study
    print("\n[Step 6/7] Running HMM & Log-Domain Viterbi Route Reconstruction on Student-Test Trips...")
    hmm_cfg = HMMConfig()
    tuned_test_probs = tuning_summary["tuned_test_probs"]

    hmm_rows = []
    offset = 0
    total_clf_hits = 0
    total_hmm_hits = 0
    total_obs_count = 0
    all_hmm_wta_preds = []
    all_clf_wta_preds = []

    for trip in test_trips:
        cands = test_cands[trip.trip_id]
        n_obs, k = cands.cand_arcs.shape
        n_rows = n_obs * k
        trip_probs = tuned_test_probs[offset : offset + n_rows].reshape(n_obs, k)
        offset += n_rows

        # Classifier-only pointwise predictions
        clf_best_k = np.argmax(trip_probs, axis=1)
        clf_pred_arcs = cands.cand_arcs[np.arange(n_obs), clf_best_k]
        clf_cont = compute_route_continuity(trip, clf_pred_arcs)

        clf_dedup_route = [int(clf_pred_arcs[0])]
        for a in clf_pred_arcs[1:]:
            if int(a) != clf_dedup_route[-1]:
                clf_dedup_route.append(int(a))
        clf_seq_metrics = compute_sequence_route_metrics(clf_dedup_route, trip.route)

        # HMM + Viterbi decoding
        t0_vit = time.perf_counter()
        hmm_pred_arcs, hmm_reconstructed_route, hmm_stats = viterbi_decode_trip(
            trip, cands, trip_probs, config=hmm_cfg
        )
        vit_ms = (time.perf_counter() - t0_vit) * 1000.0
        hmm_seq_metrics = compute_sequence_route_metrics(hmm_reconstructed_route, trip.route)

        # Map HMM predicted arc back to candidate binary assignment for Macro F1 calculation
        clf_wta = np.zeros((n_obs, k), dtype=np.int32)
        clf_wta[np.arange(n_obs), clf_best_k] = 1
        all_clf_wta_preds.append(clf_wta.ravel())

        hmm_wta = (cands.cand_arcs == hmm_pred_arcs[:, None]).astype(np.int32)
        all_hmm_wta_preds.append(hmm_wta.ravel())

        clf_hits = int(np.sum(clf_pred_arcs == trip.obs_ref_arcs))
        hmm_hits = int(np.sum(hmm_pred_arcs == trip.obs_ref_arcs))
        total_clf_hits += clf_hits
        total_hmm_hits += hmm_hits
        total_obs_count += n_obs

        clf_top1 = float(clf_hits / n_obs)
        hmm_top1 = float(hmm_hits / n_obs)

        trip_y = cands.labels.ravel()
        clf_trip_f1 = float(f1_score(trip_y, clf_wta.ravel(), average="macro", zero_division=0))
        hmm_trip_f1 = float(f1_score(trip_y, hmm_wta.ravel(), average="macro", zero_division=0))

        hmm_rows.append(
            {
                "trip_id": trip.trip_id,
                "n_observations": n_obs,
                "n_ref_route_arcs": len(trip.route) if trip.route is not None else 0,
                "candidate_miss_rate": cands.candidate_miss_rate,
                "clf_top1_accuracy": clf_top1,
                "hmm_top1_accuracy": hmm_top1,
                "top1_gain_pct": (hmm_top1 - clf_top1) * 100.0,
                "clf_macro_f1": clf_trip_f1,
                "hmm_macro_f1": hmm_trip_f1,
                "clf_route_continuity": clf_cont["connected_continuity"],
                "hmm_route_continuity": hmm_stats["route_continuity"],
                "clf_strict_1hop_continuity": clf_cont["strict_1hop_continuity"],
                "hmm_strict_1hop_continuity": hmm_stats["strict_1hop_continuity"],
                "clf_seq_f1": clf_seq_metrics["route_seq_f1"],
                "hmm_seq_f1": hmm_seq_metrics["route_seq_f1"],
                "viterbi_decode_ms": vit_ms,
            }
        )

        # Save predicted .route and observation-level predictions
        pred_route_path = os.path.join(PREDICTIONS_OUT_DIR, f"{trip.trip_id}.route")
        np.savetxt(pred_route_path, np.array(hmm_reconstructed_route, dtype=np.int32), fmt="%d")
        pred_obs_path = os.path.join(PREDICTIONS_OUT_DIR, f"{trip.trip_id}_obs_arcs.txt")
        np.savetxt(pred_obs_path, hmm_pred_arcs, fmt="%d")

        # Generate Folium interactive maps & static report figures for representative trips
        if trip.trip_id in ("00000000", "00000007", "00000052"):
            html_path = os.path.join(MAPS_OUT_DIR, f"folium_trip_{trip.trip_id}.html")
            generate_folium_map_for_trip(
                trip,
                clf_only_arcs=clf_pred_arcs,
                hmm_pred_arcs=hmm_pred_arcs,
                hmm_reconstructed_route=hmm_reconstructed_route,
                out_html_path=html_path,
            )
            png_path = os.path.join(MAPS_OUT_DIR, f"map_comparison_{trip.trip_id}.png")
            generate_static_map_figure_for_trip(
                trip,
                clf_only_arcs=clf_pred_arcs,
                hmm_pred_arcs=hmm_pred_arcs,
                hmm_reconstructed_route=hmm_reconstructed_route,
                out_png_path=png_path,
            )

    hmm_df = pd.DataFrame(hmm_rows)
    hmm_df.to_csv(os.path.join(RESULTS_OUT_DIR, "hmm_viterbi_ablation.csv"), index=False)

    overall_clf_f1 = float(f1_score(y_test, np.concatenate(all_clf_wta_preds), average="macro", zero_division=0))
    overall_hmm_f1 = float(f1_score(y_test, np.concatenate(all_hmm_wta_preds), average="macro", zero_division=0))
    overall_clf_top1 = float(total_clf_hits / max(1, total_obs_count))
    overall_hmm_top1 = float(total_hmm_hits / max(1, total_obs_count))

    hmm_summary = {
        "overall_clf_top1_accuracy": overall_clf_top1,
        "overall_hmm_top1_accuracy": overall_hmm_top1,
        "overall_clf_macro_f1": overall_clf_f1,
        "overall_hmm_macro_f1": overall_hmm_f1,
        "mean_clf_route_continuity": float(hmm_df["clf_route_continuity"].mean()),
        "mean_hmm_route_continuity": float(hmm_df["hmm_route_continuity"].mean()),
        "mean_clf_strict_1hop_continuity": float(hmm_df["clf_strict_1hop_continuity"].mean()),
        "mean_hmm_strict_1hop_continuity": float(hmm_df["hmm_strict_1hop_continuity"].mean()),
        "mean_clf_seq_f1": float(hmm_df["clf_seq_f1"].mean()),
        "mean_hmm_seq_f1": float(hmm_df["hmm_seq_f1"].mean()),
        "mean_viterbi_decode_ms": float(hmm_df["viterbi_decode_ms"].mean()),
    }
    with open(os.path.join(RESULTS_OUT_DIR, "hmm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(hmm_summary, f, indent=2)

    print(
        f"  -> Classifier-Only: Top-1 Acc = {overall_clf_top1*100:.2f}% | "
        f"Macro F1 = {overall_clf_f1:.4f} | Continuity = {hmm_summary['mean_clf_route_continuity']*100:.2f}%"
    )
    print(
        f"  -> Classifier+HMM:  Top-1 Acc = {overall_hmm_top1*100:.2f}% | "
        f"Macro F1 = {overall_hmm_f1:.4f} | Continuity = {hmm_summary['mean_hmm_route_continuity']*100:.2f}%"
    )

    # Also generate an EDA Folium map for Training Trip 00000001
    if len(train_trips) > 0:
        sample_tr = train_trips[0]
        generate_folium_map_for_trip(
            sample_tr,
            out_html_path=os.path.join(MAPS_OUT_DIR, f"eda_train_trip_{sample_tr.trip_id}.html"),
        )

    # 7. Generate All Plots & Save Final Model Artifacts
    print("\n[Step 7/7] Generating Compulsory Visualisations & Saving Model Artifacts...")
    plot_model_comparisons(df_models, out_dir=FIGURES_OUT_DIR)
    plot_confusion_matrices(diagnostics, out_dir=FIGURES_OUT_DIR)
    plot_epoch_loss_curves(diagnostics, out_dir=FIGURES_OUT_DIR)
    plot_bayesian_optimisation(trials_df, tuned_comp_df, out_dir=FIGURES_OUT_DIR)
    plot_hmm_ablation(hmm_df, out_dir=FIGURES_OUT_DIR)
    plot_feature_importance_and_shap(best_model, X_test, out_dir=FIGURES_OUT_DIR)

    # Save best model and all comparison models
    joblib.dump(best_model, os.path.join(MODELS_DIR, "pathsnap_best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "hmm_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(hmm_cfg), f, indent=2)

    metadata = {
        "team_info": TEAM_INFO,
        "random_seed": RANDOM_SEED,
        "selected_model_name": "Bayesian-Tuned LightGBM",
        "feature_names": FEATURE_NAMES,
        "top_k_candidates": TOP_K_CANDIDATES,
        "best_hyperparameters": tuning_summary["best_params"],
        "best_cv_macro_f1": tuning_summary["best_cv_macro_f1"],
        "hmm_summary": hmm_summary,
        "eda_summary": eda_stats,
        "candidate_miss_rate_train": mean_train_miss,
        "candidate_miss_rate_student_test": mean_test_miss,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"TRAINING & EVALUATION COMPLETE in {total_elapsed:.1f}s!")
    print(f"Saved model: {os.path.join(MODELS_DIR, 'pathsnap_best_model.joblib')}")
    print(f"Saved results: {RESULTS_OUT_DIR}")
    print(f"Saved figures: {FIGURES_OUT_DIR}")
    print(f"Saved Folium maps: {MAPS_OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
