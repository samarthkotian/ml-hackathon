"""
Comprehensive Exploratory Data Analysis (EDA) and Data Quality Audit
for PathSnap (Team SI No 107).
"""
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import EDA_OUT_DIR
from src.data_loader import TripData


def run_eda_and_save_artifacts(
    train_trips: List[TripData],
    test_trips: List[TripData],
    out_dir: str = EDA_OUT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Inspect all training and student-testing trips, audit missing values,
    sampling intervals, coordinate bounds, road-network connectivity, and
    save summary CSVs and multi-panel EDA visualisations.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_trips = train_trips + test_trips

    records = []
    all_speeds = []
    all_dts = []
    all_ref_dists = []
    all_arc_lengths = []
    all_node_degrees = []

    for trip in all_trips:
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs
        route = trip.route

        n_obs = len(track)
        n_nodes = len(nodes)
        n_arcs = len(arcs)
        n_route = len(route) if route is not None else 0

        # Missing or invalid values check
        missing_track = int(np.isnan(track).sum() + np.isinf(track).sum())
        missing_nodes = int(np.isnan(nodes).sum() + np.isinf(nodes).sum())
        missing_arcs = int(np.isnan(arcs).sum() + np.isinf(arcs).sum())

        # Sampling intervals dt
        dt = np.diff(track[:, 2]) if n_obs > 1 else np.array([1.0])
        dt_eq_1_pct = float(np.mean(np.isclose(dt, 1.0, atol=1e-2)) * 100.0)
        max_dt = float(np.max(dt)) if len(dt) > 0 else 1.0
        min_dt = float(np.min(dt)) if len(dt) > 0 else 1.0
        n_gaps_gt_5s = int(np.sum(dt > 5.0))
        is_monotonic_ts = bool(np.all(dt > 0))

        # Trajectory length and speeds
        step_d = np.hypot(np.diff(trip.track_x), np.diff(trip.track_y)) if n_obs > 1 else np.array([0.0])
        traj_len_km = float(np.sum(step_d) / 1000.0)
        speeds = step_d / np.maximum(dt, 1e-3)

        # Road-network connectivity & isolated nodes
        total_deg = trip.node_in_degree + trip.node_out_degree
        n_isolated_nodes = int(np.sum(total_deg == 0))
        n_dead_end_nodes = int(np.sum(trip.node_out_degree == 0))

        # Reference route connectivity
        route_conn_ok = 0
        if route is not None and len(route) > 1:
            for i in range(len(route) - 1):
                a1, a2 = int(route[i]), int(route[i + 1])
                if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                    route_conn_ok += 1
            route_conn_pct = float(route_conn_ok / (len(route) - 1) * 100.0)
        else:
            route_conn_pct = 100.0

        median_gps_noise_m = (
            float(np.median(trip.obs_ref_dists))
            if trip.obs_ref_dists is not None and len(trip.obs_ref_dists) > 0
            else np.nan
        )
        p95_gps_noise_m = (
            float(np.percentile(trip.obs_ref_dists, 95))
            if trip.obs_ref_dists is not None and len(trip.obs_ref_dists) > 0
            else np.nan
        )

        records.append(
            {
                "trip_id": trip.trip_id,
                "partition": trip.partition,
                "n_observations": n_obs,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "n_route_segments": n_route,
                "missing_values": missing_track + missing_nodes + missing_arcs,
                "timestamps_monotonic": is_monotonic_ts,
                "pct_1hz_sampling": dt_eq_1_pct,
                "max_time_gap_s": max_dt,
                "n_gaps_gt_5s": n_gaps_gt_5s,
                "trajectory_length_km": traj_len_km,
                "mean_speed_mps": float(np.mean(speeds)),
                "ref_lon": trip.ref_lon,
                "ref_lat": trip.ref_lat,
                "isolated_nodes": n_isolated_nodes,
                "dead_end_nodes": n_dead_end_nodes,
                "route_connected_pct": route_conn_pct,
                "median_gps_error_m": median_gps_noise_m,
                "p95_gps_error_m": p95_gps_noise_m,
            }
        )

        # Collect samples for distribution plots
        all_speeds.append(speeds[speeds < 55.0])
        all_dts.append(dt)
        if trip.obs_ref_dists is not None:
            all_ref_dists.append(trip.obs_ref_dists[trip.obs_ref_dists < 80.0])
        all_arc_lengths.append(trip.arc_lengths[trip.arc_lengths < 600.0])
        all_node_degrees.append(total_deg[total_deg > 0])

    inv_df = pd.DataFrame(records)
    inv_df.to_csv(os.path.join(out_dir, "dataset_inventory.csv"), index=False)

    # Build partition summary table
    part_rows = []
    for part_name, grp in inv_df.groupby("partition", sort=False):
        part_rows.append(
            {
                "partition": part_name,
                "n_trips": int(len(grp)),
                "total_gps_obs": int(grp["n_observations"].sum()),
                "mean_gps_obs_per_trip": float(grp["n_observations"].mean()),
                "median_gps_obs_per_trip": float(grp["n_observations"].median()),
                "mean_nodes_per_trip": float(grp["n_nodes"].mean()),
                "mean_arcs_per_trip": float(grp["n_arcs"].mean()),
                "mean_route_len": float(grp["n_route_segments"].mean()),
                "total_missing_values": int(grp["missing_values"].sum()),
                "mean_1hz_pct": float(grp["pct_1hz_sampling"].mean()),
                "trips_with_gaps_gt_5s": int((grp["n_gaps_gt_5s"] > 0).sum()),
                "mean_route_connected_pct": float(grp["route_connected_pct"].mean()),
                "median_gps_noise_m": float(grp["median_gps_error_m"].median()),
                "p95_gps_noise_m": float(grp["p95_gps_error_m"].median()),
            }
        )
    part_df = pd.DataFrame(part_rows)
    part_df.to_csv(os.path.join(out_dir, "partition_summary.csv"), index=False)

    # Plot 1: Dataset & Trajectory Overview (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=220)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    train_sub = inv_df[inv_df["partition"] == "train"]
    test_sub = inv_df[inv_df["partition"] == "student_test"]

    # Panel A: GPS Observations per Trip
    ax = axes[0, 0]
    ax.hist(
        train_sub["n_observations"],
        bins=22,
        alpha=0.78,
        color="#2563eb",
        edgecolor="white",
        label=f"Train (n={len(train_sub)} trips)",
    )
    ax.hist(
        test_sub["n_observations"],
        bins=10,
        alpha=0.85,
        color="#f59e0b",
        edgecolor="white",
        label=f"Student Test (n={len(test_sub)} trips)",
    )
    ax.set_title("(a) GPS Observations per Trip by Partition", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of GPS Observations (.track rows)")
    ax.set_ylabel("Number of Trips")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # Panel B: Road Segments (.arcs) vs Nodes (.nodes)
    ax = axes[0, 1]
    ax.scatter(
        train_sub["n_nodes"],
        train_sub["n_arcs"],
        c="#2563eb",
        alpha=0.7,
        s=45,
        label="Train Trips (80)",
    )
    ax.scatter(
        test_sub["n_nodes"],
        test_sub["n_arcs"],
        c="#dc2626",
        marker="D",
        s=55,
        label="Student Test Trips (10)",
    )
    ax.set_title("(b) Road Network Size: Directed Arcs vs Nodes", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Road Nodes (.nodes)")
    ax.set_ylabel("Number of Directed Road Segments (.arcs)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # Panel C: Vehicle Speed Distribution
    ax = axes[1, 0]
    speeds_concat = np.concatenate(all_speeds)
    ax.hist(speeds_concat, bins=45, color="#10b981", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(speeds_concat), color="#065f46", linestyle="--", linewidth=1.8,
               label=f"Median: {np.median(speeds_concat):.1f} m/s ({np.median(speeds_concat)*3.6:.1f} km/h)")
    ax.set_title("(c) Instantaneous Vehicle Speed Distribution (m/s)", fontsize=11, fontweight="bold")
    ax.set_xlabel("GPS Instantaneous Speed (m/s)")
    ax.set_ylabel("Observation Count")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # Panel D: Sampling Interval Irregularities (dt > 1s gaps)
    ax = axes[1, 1]
    dts_concat = np.concatenate(all_dts)
    irregular_dts = dts_concat[(dts_concat > 1.05) & (dts_concat <= 60.0)]
    pct_exact_1s = float(np.mean(np.isclose(dts_concat, 1.0, atol=0.05)) * 100.0)
    ax.hist(irregular_dts, bins=30, color="#8b5cf6", edgecolor="white", alpha=0.85)
    ax.set_title(
        f"(d) Irregular GPS Time Gaps (1 Hz Nominal = {pct_exact_1s:.1f}% of Steps)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Time Gap dt > 1s (clipped at 60s for visibility)")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.suptitle(
        "PathSnap Exploratory Data Analysis — Trajectory & Sampling Characteristics (Train + Student-Test)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "eda_dataset_overview.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Spatial Noise, Segment Length, Node Degree & Geographical Distribution
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=220)

    # Panel A: Orthogonal GPS Noise Distribution relative to Reference Route
    ax = axes[0, 0]
    noise_concat = np.concatenate(all_ref_dists)
    ax.hist(noise_concat, bins=45, color="#ef4444", edgecolor="white", alpha=0.82)
    med_n = np.median(noise_concat)
    p95_n = np.percentile(noise_concat, 95)
    ax.axvline(med_n, color="#7f1d1d", linestyle="--", linewidth=1.8, label=f"Median Error: {med_n:.2f} m")
    ax.axvline(p95_n, color="#b91c1c", linestyle=":", linewidth=1.8, label=f"95th Percentile: {p95_n:.2f} m")
    ax.set_title("(a) Raw GPS Orthogonal Error to Reference Route (m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Perpendicular Distance from GPS Point to True Segment (m)")
    ax.set_ylabel("Observation Count")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # Panel B: Road Segment Length Distribution
    ax = axes[0, 1]
    lens_concat = np.concatenate(all_arc_lengths)
    ax.hist(lens_concat, bins=45, color="#0ea5e9", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(lens_concat), color="#0369a1", linestyle="--", linewidth=1.8,
               label=f"Median Length: {np.median(lens_concat):.1f} m")
    ax.set_title("(b) Road Network Directed Segment Length Distribution (m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Road Segment Length (m)")
    ax.set_ylabel("Number of Segments")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    # Panel C: Road Graph Node Total Degree Distribution
    ax = axes[1, 0]
    deg_concat = np.concatenate(all_node_degrees)
    deg_vals, deg_counts = np.unique(np.clip(deg_concat, 1, 10), return_counts=True)
    ax.bar(deg_vals, deg_counts / len(deg_concat) * 100.0, color="#14b8a6", edgecolor="white", width=0.65)
    ax.set_title("(c) Road-Network Node Degree Distribution (In + Out Degree)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Total Node Degree (clipped at 10+)")
    ax.set_ylabel("Percentage of Active Road Nodes (%)")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)

    # Panel D: Geographical Centroids of Trips
    ax = axes[1, 1]
    ax.scatter(
        train_sub["ref_lon"],
        train_sub["ref_lat"],
        c="#2563eb",
        alpha=0.75,
        s=50,
        label="Train Trips (80)",
    )
    ax.scatter(
        test_sub["ref_lon"],
        test_sub["ref_lat"],
        c="#f97316",
        marker="^",
        s=75,
        edgecolor="black",
        linewidth=0.6,
        label="Student Test Trips (10)",
    )
    ax.set_title("(d) Geographical Distribution of Trip Centroids (WGS84)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "PathSnap Exploratory Data Analysis — Spatial Error & Road Network Topology",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "eda_spatial_and_network.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)

    summary_stats = {
        "n_train_trips": len(train_trips),
        "n_student_test_trips": len(test_trips),
        "total_train_obs": int(train_sub["n_observations"].sum()),
        "total_test_obs": int(test_sub["n_observations"].sum()),
        "pct_exact_1s": pct_exact_1s,
        "median_gps_error_m": float(med_n),
        "p95_gps_error_m": float(p95_n),
        "median_segment_len_m": float(np.median(lens_concat)),
        "fig1_path": fig1_path,
        "fig2_path": fig2_path,
    }
    return inv_df, part_df, summary_stats
