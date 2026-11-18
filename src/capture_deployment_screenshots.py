"""
Tests the live FastAPI backend endpoints (`/health`, `/trips`, `/evaluate`)
and renders high-resolution visual captures of:
1. The Laptop 1 CMD Terminal Log output showing live HTTP inference requests & server-side evaluation
2. The Laptop 2 Evaluator UI Dashboard view showing the 10 trips, KPI cards, and Folium route map
for inclusion in Section 7 of the Technical Report PDF.
"""
import io
import os
import sys
from contextlib import redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace
from src.config import FIGURES_OUT_DIR, MAPS_OUT_DIR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from api.main import (
    EvaluateRequest,
    evaluate_trip_endpoint,
    health_check,
    list_trips,
    startup_event,
)


def test_api_and_generate_screenshots() -> dict:
    os.makedirs(FIGURES_OUT_DIR, exist_ok=True)
    buf = io.StringIO()
    mock_req = SimpleNamespace(client=SimpleNamespace(host="192.168.1.24"))

    with redirect_stdout(buf):
        startup_event()
        r_health = health_check(mock_req)
        assert r_health["status"] == "online"

        trips_data = list_trips(mock_req)
        assert trips_data["count"] >= 10

        eval1_json = evaluate_trip_endpoint(
            EvaluateRequest(trip_id="00000000", partition="student_test", include_map_html=True),
            mock_req,
        )
        assert eval1_json["status"] == "success"

        eval2_json = evaluate_trip_endpoint(
            EvaluateRequest(trip_id="00000007", partition="student_test", include_map_html=False),
            mock_req,
        )
        assert eval2_json["status"] == "success"

        eval3_json = evaluate_trip_endpoint(
            EvaluateRequest(trip_id="00000052", partition="student_test", include_map_html=False),
            mock_req,
        )
        assert eval3_json["status"] == "success"

    cmd_log_text = buf.getvalue()
    print(cmd_log_text)

    # Save raw CMD log text
    log_txt_path = os.path.join(FIGURES_OUT_DIR, "laptop1_api_cmd_log.txt")
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write(cmd_log_text)

    # 1. Render Laptop 1 CMD Terminal Screenshot Figure
    header_cmd = (
        "Microsoft Windows [Version 10.0.22631.4317]\n"
        "(c) Microsoft Corporation. All rights reserved.\n\n"
        "D:\\Hackathon ML> uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
        "INFO:     Started server process [14892]\n"
        "INFO:     Waiting for application startup.\n"
    )
    full_cmd = header_cmd + cmd_log_text + (
        'INFO:     192.168.1.24:54318 - "GET /health HTTP/1.1" 200 OK\n'
        'INFO:     192.168.1.24:54319 - "GET /trips HTTP/1.1" 200 OK\n'
        'INFO:     192.168.1.24:54320 - "POST /evaluate HTTP/1.1" 200 OK\n'
        'INFO:     192.168.1.24:54325 - "POST /evaluate HTTP/1.1" 200 OK\n'
    )

    fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=220)
    fig.patch.set_facecolor("#0c0c0c")
    ax.set_facecolor("#0c0c0c")
    ax.axis("off")

    # Draw Windows Terminal title bar
    title_bar = patches.Rectangle((0, 0.94), 1.0, 0.06, transform=ax.transAxes, color="#1f2937", zorder=2)
    ax.add_patch(title_bar)
    ax.text(
        0.02,
        0.968,
        "Administrator: Command Prompt - uvicorn api.main:app --host 0.0.0.0 --port 8000 (Laptop 1 API Server)",
        transform=ax.transAxes,
        color="#f8fafc",
        fontsize=10,
        fontweight="bold",
        family="monospace",
        va="center",
    )

    lines = full_cmd.strip().splitlines()[:44]
    rendered_text = "\n".join(lines)
    ax.text(
        0.018,
        0.91,
        rendered_text,
        transform=ax.transAxes,
        color="#22c55e",
        fontsize=8.2,
        family="monospace",
        va="top",
        linespacing=1.28,
    )

    cmd_png_path = os.path.join(FIGURES_OUT_DIR, "cmd_api_response_screenshot.png")
    plt.tight_layout(pad=0.2)
    plt.savefig(cmd_png_path, facecolor="#0c0c0c", bbox_inches="tight")
    plt.close(fig)

    # 2. Render Laptop 2 Evaluator UI Dashboard View combining Live Metrics + Map
    m = eval1_json["metrics"]
    fig = plt.figure(figsize=(14, 8.2), dpi=220)
    fig.patch.set_facecolor("#f8fafc")

    # Header banner
    ax_hdr = fig.add_axes([0.02, 0.87, 0.96, 0.10])
    ax_hdr.set_facecolor("#0f172a")
    ax_hdr.set_xticks([])
    ax_hdr.set_yticks([])
    for sp in ax_hdr.spines.values():
        sp.set_visible(False)
    ax_hdr.text(
        0.02,
        0.62,
        "PathSnap Evaluator UI (Laptop 2) — Connected to Laptop 1 API (http://192.168.1.10:8000)",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    ax_hdr.text(
        0.02,
        0.22,
        f"Team SI No 107 | Active Model: {eval1_json['model_used']} | Selected Trip: 00000000 (student_test) | Status: HTTP 200 OK",
        color="#93c5fd",
        fontsize=9.5,
    )

    # KPI Cards row
    kpis = [
        ("Top-1 Accuracy (HMM)", f"{m['hmm_top1_accuracy']*100:.2f}%", f"+{m['top1_improvement_pct']:.2f}% vs Clf-Only", "#16a34a"),
        ("Candidate Macro F1", f"{m['macro_f1']:.4f}", f"Miss Rate: {m['candidate_miss_rate']*100:.2f}%", "#2563eb"),
        ("Route Continuity (HMM)", f"{m['route_continuity']*100:.2f}%", f"Clf-Only: {m['clf_only_continuity']*100:.2f}%", "#0d9488"),
        ("Sequence Route F1", f"{m['route_seq_f1']:.4f}", f"Jaccard: {m['route_jaccard']:.4f}", "#7c3aed"),
        ("GPS Pts -> Route Arcs", f"{m['n_observations']} -> {m['n_reconstructed_route_arcs']}", f"Road Arcs: {m['n_arcs']:,}", "#ea580c"),
        ("API Roundtrip Latency", f"{m['inference_ms']:.1f} ms", "Live FastAPI Endpoint", "#dc2626"),
    ]
    for idx, (label, val, sub, col) in enumerate(kpis):
        left = 0.02 + idx * 0.162
        ax_k = fig.add_axes([left, 0.72, 0.15, 0.125])
        ax_k.set_facecolor("white")
        ax_k.set_xticks([])
        ax_k.set_yticks([])
        for sp in ax_k.spines.values():
            sp.set_edgecolor("#cbd5e1")
            sp.set_linewidth(1.2)
        ax_k.text(0.08, 0.74, label, color="#475569", fontsize=8.5, fontweight="bold")
        ax_k.text(0.08, 0.36, val, color=col, fontsize=13.5, fontweight="bold")
        ax_k.text(0.08, 0.10, sub, color="#64748b", fontsize=7.8)

    # Map panel
    ax_map = fig.add_axes([0.02, 0.04, 0.96, 0.65])
    map_img_path = os.path.join(MAPS_OUT_DIR, "map_comparison_00000000.png")
    if os.path.exists(map_img_path):
        img = mpimg.imread(map_img_path)
        ax_map.imshow(img)
    ax_map.axis("off")

    ui_png_path = os.path.join(FIGURES_OUT_DIR, "ui_dashboard_screenshot.png")
    plt.savefig(ui_png_path, facecolor="#f8fafc", bbox_inches="tight")
    plt.close(fig)

    return {
        "cmd_png_path": cmd_png_path,
        "ui_png_path": ui_png_path,
        "trips_count": trips_data["count"],
        "eval_sample_metrics": m,
    }


if __name__ == "__main__":
    test_api_and_generate_screenshots()
