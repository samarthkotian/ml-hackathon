"""
Generates:
1. A comprehensive 4-panel Master Dashboard Graph comparing all 7 models + Bayesian-Tuned + HMM/Viterbi
   (`outputs/figures/all_models_master_dashboard.png`)
2. Dedicated 3-panel diagnostic graphs for EVERY individual model (`outputs/figures/per_model/01_...` to `07_...`)
   showing genuine 5-Fold CV fold-by-fold stability, Train vs CV vs Student-Test generalisation, and Confusion Matrix.
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import FIGURES_OUT_DIR, RESULTS_OUT_DIR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_graphs() -> None:
    os.makedirs(FIGURES_OUT_DIR, exist_ok=True)
    per_model_dir = os.path.join(FIGURES_OUT_DIR, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_OUT_DIR, "model_comparison_results.csv"))
    bo_df = pd.read_csv(os.path.join(RESULTS_OUT_DIR, "bayesian_untuned_vs_tuned.csv"))
    with open(os.path.join(RESULTS_OUT_DIR, "model_diagnostics.json"), "r", encoding="utf-8") as f:
        diag = json.load(f)
    with open(os.path.join(RESULTS_OUT_DIR, "hmm_summary.json"), "r", encoding="utf-8") as f:
        hmm_sum = json.load(f)

    # =========================================================================
    # 1. MASTER 4-PANEL DASHBOARD GRAPH FOR ALL MODELS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=240)
    models = df["model_name"].tolist()
    x = np.arange(len(models))
    w = 0.26

    # Panel A: Train vs 5-Fold CV vs Student-Test Macro F1 (Overfitting/Generalisation Check)
    ax = axes[0, 0]
    b_tr = ax.bar(x - w, df["train_macro_f1"], width=w, color="#94a3b8", label="Train Macro F1")
    b_cv = ax.bar(x, df["cv_macro_f1_mean"], width=w, yerr=df["cv_macro_f1_std"], capsize=3, color="#2563eb", label="5-Fold GroupCV Macro F1")
    b_te = ax.bar(x + w, df["student_test_macro_f1"], width=w, color="#16a34a", label="Student-Test Macro F1")
    ax.set_title("(a) Macro F1-Score: Train vs 5-Fold GroupCV vs Student-Test", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("Macro F1-Score")
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    for bar in b_te:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Panel B: 5-Fold Cross-Validation Fold-by-Fold Stability Curve for All 7 Models
    ax = axes[0, 1]
    palette = ["#64748b", "#f97316", "#10b981", "#14b8a6", "#8b5cf6", "#2563eb", "#dc2626"]
    folds = np.arange(1, 6)
    for idx, mname in enumerate(models):
        f_scores = diag[mname]["fold_f1s"]
        ax.plot(folds, f_scores, marker="o", linewidth=2.0, color=palette[idx % len(palette)], label=f"{mname} (mean={np.mean(f_scores):.3f})")
    ax.set_title("(b) Fold-by-Fold Macro F1 Stability Across 5 GroupKFold Splits", fontsize=11, fontweight="bold")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("Validation Fold Macro F1-Score")
    ax.legend(frameon=True, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel C: Full Pipeline Progression (All 7 Models -> Bayesian-Tuned -> +HMM/Viterbi)
    ax = axes[1, 0]
    prog_names = models + ["Bayesian-Tuned\nLightGBM", "Tuned LightGBM\n+ HMM/Viterbi"]
    prog_top1 = list(df["student_test_top1_accuracy"] * 100.0) + [
        float(bo_df.iloc[1]["student_test_top1_accuracy"]) * 100.0,
        float(hmm_sum["overall_hmm_top1_accuracy"]) * 100.0,
    ]
    prog_colors = ["#64748b"] * 5 + ["#2563eb", "#0284c7", "#059669", "#dc2626"]
    xp = np.arange(len(prog_names))
    bars_p = ax.bar(xp, prog_top1, width=0.58, color=prog_colors, edgecolor="white")
    ax.set_title("(c) End-to-End Top-1 Accuracy Progression Across All 7 Models + Tuning + HMM", fontsize=11, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(prog_names, rotation=24, ha="right", fontsize=8.5)
    ax.set_ylim(80.0, 95.0)
    ax.set_ylabel("Observation-Level Top-1 Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars_p:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel D: Multi-Metric Radar / Tradeoff Scatter (Accuracy vs Log Loss vs Continuity)
    ax = axes[1, 1]
    sizes = np.clip(400.0 / np.sqrt(df["inference_ms_per_trip"]), 60, 260)
    sc = ax.scatter(
        df["student_test_log_loss"],
        df["student_test_top1_accuracy"] * 100.0,
        s=sizes,
        c=df["route_continuity"] * 100.0,
        cmap="viridis",
        edgecolors="black",
        linewidth=1.2,
        zorder=4,
    )
    for _, r in df.iterrows():
        ax.annotate(
            r["model_name"],
            (r["student_test_log_loss"], r["student_test_top1_accuracy"] * 100.0),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Route Continuity (%)")
    ax.set_title("(d) Accuracy vs Probability Log Loss (Bubble Size = Faster Inference)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Student-Test Binary Log Loss (Lower / Left is Better)")
    ax.set_ylabel("Student-Test Top-1 Accuracy (%) (Higher is Better)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "PathSnap Master Model Comparison Dashboard — All 7 Models, Bayesian Tuning & HMM/Viterbi",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    master_path = os.path.join(FIGURES_OUT_DIR, "all_models_master_dashboard.png")
    plt.savefig(master_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved master dashboard: {master_path}")

    # =========================================================================
    # 2. INDIVIDUAL DEDICATED 3-PANEL GRAPH FOR EVERY SINGLE MODEL (1 to 7)
    # =========================================================================
    for idx, r in df.iterrows():
        mname = r["model_name"]
        safe_name = f"0{idx+1}_{mname.replace(' ', '_').replace('-', '_')}.png"
        d = diag[mname]

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=220)

        # Panel 1: 5-Fold GroupKFold Fold Stability (or Iterative Epoch Curve if available)
        ax1 = axes[0]
        if d.get("epoch_history") is not None:
            hist = d["epoch_history"]
            tr_l = hist.get("train_loss", [])
            va_l = hist.get("val_loss", [])
            ep = np.arange(1, len(tr_l) + 1)
            ax1.plot(ep, tr_l, color="#2563eb", linewidth=2.0, label="Train Log Loss")
            if va_l:
                ax1.plot(ep, va_l, color="#dc2626", linestyle="--", linewidth=2.0, label="Val Log Loss")
            ax1.set_title(f"(a) {mname}: Iterative Loss Curve", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Epoch / Boosting Iteration")
            ax1.set_ylabel("Binary Log Loss")
            ax1.legend(frameon=True, fontsize=8.5)
            ax1.grid(alpha=0.3)
        else:
            folds = np.arange(1, 6)
            ax1.plot(folds, d["fold_f1s"], marker="o", color="#2563eb", linewidth=2.0, label="Fold Macro F1")
            ax1.plot(folds, d["fold_top1s"], marker="s", color="#16a34a", linestyle="--", linewidth=2.0, label="Fold Top-1 Acc")
            ax1.axhline(np.mean(d["fold_f1s"]), color="#2563eb", alpha=0.35, linestyle=":")
            ax1.set_title(f"(a) {mname}: 5-Fold GroupCV Stability", fontsize=10, fontweight="bold")
            ax1.set_xticks(folds)
            ax1.set_xticklabels([f"Fold {i}" for i in folds])
            ax1.set_ylim(0.65, 1.0)
            ax1.set_ylabel("Score")
            ax1.legend(frameon=True, fontsize=8.5)
            ax1.grid(alpha=0.3)

        # Panel 2: Train vs CV vs Student-Test Metrics Bar Chart
        ax2 = axes[1]
        m_labels = ["Train\nMacro F1", "5-Fold CV\nMacro F1", "Test\nMacro F1", "Test\nTop-1 Acc", "Route\nContinuity"]
        m_vals = [
            float(r["train_macro_f1"]),
            float(r["cv_macro_f1_mean"]),
            float(r["student_test_macro_f1"]),
            float(r["student_test_top1_accuracy"]),
            float(r["route_continuity"]),
        ]
        m_cols = ["#94a3b8", "#2563eb", "#16a34a", "#8b5cf6", "#f59e0b"]
        bars2 = ax2.bar(np.arange(len(m_labels)), m_vals, width=0.55, color=m_cols, edgecolor="white")
        ax2.set_title(f"(b) {mname}: Evaluation Metrics", fontsize=10, fontweight="bold")
        ax2.set_xticks(np.arange(len(m_labels)))
        ax2.set_xticklabels(m_labels, fontsize=8.5)
        ax2.set_ylim(0.70, 1.05)
        ax2.set_ylabel("Score")
        ax2.grid(axis="y", alpha=0.3)
        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Panel 3: Confusion Matrix
        ax3 = axes[2]
        cm = np.array(d["confusion_matrix"])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax3.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax3.set_title(f"(c) {mname}: Confusion Matrix", fontsize=10, fontweight="bold")
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(["Pred 0 (Non-Ref)", "Pred 1 (Ref Arc)"], fontsize=8.5)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["True 0", "True 1"], fontsize=8.5)
        for ri in range(2):
            for ci in range(2):
                val = cm[ri, ci]
                pct = cm_norm[ri, ci] * 100.0
                col = "white" if cm_norm[ri, ci] > 0.55 else "#0f172a"
                ax3.text(ci, ri, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=col, fontsize=9, fontweight="bold")

        plt.suptitle(
            f"Model {idx+1}: {mname} ({r['algorithm_family']}) — Diagnostic Report",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        out_p = os.path.join(per_model_dir, safe_name)
        plt.savefig(out_p, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved individual model graph: {out_p}")


if __name__ == "__main__":
    generate_all_graphs()
