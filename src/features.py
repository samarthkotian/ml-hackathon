"""
Feature engineering for PathSnap candidate classification (Team SI No 107).
Extracts 25 spatial, geometric, kinematic, heading-alignment, topological,
and local temporal-context features for every (observation, candidate_arc) pair.
Strictly zero data leakage: uses only GPS track observations and road-network geometry.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_loader import TripData, point_to_segments_distance
from src.candidates import TripCandidates, generate_candidates_for_trip

FEATURE_NAMES: List[str] = [
    "perp_dist_m",
    "dist_rank",
    "dist_ratio_to_min",
    "dist_diff_to_min",
    "dist_zscore_in_step",
    "dist_to_second_best_gap",
    "proj_alpha",
    "endpoint_clamp_overshoot",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_length_m",
    "heading_cos",
    "heading_sin",
    "heading_abs_diff_rad",
    "smoothed_heading_cos",
    "gps_speed_mps",
    "gps_accel_mps2",
    "gps_dt_sec",
    "from_node_in_deg",
    "to_node_out_deg",
    "prev_connected_min_dist",
    "next_connected_min_dist",
    "is_prev_top1_connected",
    "is_next_top1_connected",
    "window3_mean_perp_dist",
]


def _compute_kinematics(trip: TripData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous time step dt, speed, acceleration, instantaneous bearing,
    and 5-step smoothed trajectory bearing from GPS (x, y, t).
    """
    x = trip.track_x
    y = trip.track_y
    t = trip.track[:, 2]
    n = len(x)

    if n == 1:
        return (
            np.ones(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )

    dt = np.ones(n, dtype=np.float64)
    dt[1:] = np.maximum(t[1:] - t[:-1], 1e-3)
    dt[0] = dt[1]

    # Central difference for direction & velocity where possible
    dx = np.gradient(x)
    dy = np.gradient(y)
    step_dist = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    step_dist[0] = np.hypot(x[1] - x[0], y[1] - y[0])
    speed = step_dist / dt
    accel = np.gradient(speed) / dt

    # Use spatial-chord travel bearing for smoothed_heading and stabilize low-speed (<2 m/s) inst_bearing
    sm_bearing = trip.track_chord_bearing
    raw_inst_bearing = np.arctan2(dy, dx)
    inst_bearing = np.where(speed >= 2.0, raw_inst_bearing, sm_bearing)

    return dt, speed, accel, inst_bearing, sm_bearing


def extract_features_for_trip(
    trip: TripData, candidates: Optional[TripCandidates] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], TripCandidates]:
    """
    Extract feature matrix X of shape (N_obs * K, n_features) and optional binary
    target vector y of shape (N_obs * K,) for a single TripData object.
    """
    if candidates is None:
        candidates = generate_candidates_for_trip(trip)

    cand_arcs = candidates.cand_arcs          # (N, K)
    cand_dists = candidates.cand_dists        # (N, K)
    cand_alpha = candidates.cand_alpha        # (N, K)
    cand_raw_alpha = candidates.cand_raw_alpha  # (N, K)
    n_obs, k = cand_arcs.shape

    dt, speed, accel, inst_bearing, sm_bearing = _compute_kinematics(trip)

    # 1. Distance and within-step relative distance features
    perp_dist_m = cand_dists
    dist_rank = np.broadcast_to(np.arange(k, dtype=np.float64)[None, :], (n_obs, k))
    min_dist = cand_dists[:, [0]]
    second_dist = cand_dists[:, [min(1, k - 1)]]

    dist_ratio_to_min = cand_dists / (min_dist + 0.5)
    dist_diff_to_min = cand_dists - min_dist
    mean_d = np.mean(cand_dists, axis=1, keepdims=True)
    std_d = np.std(cand_dists, axis=1, keepdims=True) + 1e-3
    dist_zscore_in_step = (cand_dists - mean_d) / std_d
    # Positive for rank 0 (how much closer rank 0 is than rank 1), negative for others
    dist_to_second_best_gap = np.where(
        dist_rank == 0, second_dist - cand_dists, min_dist - cand_dists
    )

    # 2. Segment projection and endpoint geometry
    proj_alpha = cand_alpha
    endpoint_clamp_overshoot = np.abs(cand_raw_alpha - cand_alpha)

    u_nodes = trip.arcs[cand_arcs, 0]
    v_nodes = trip.arcs[cand_arcs, 1]

    ux = trip.nodes_x[u_nodes]
    uy = trip.nodes_y[u_nodes]
    vx = trip.nodes_x[v_nodes]
    vy = trip.nodes_y[v_nodes]

    tx = trip.track_x[:, None]
    ty = trip.track_y[:, None]

    dist_to_from_node_m = np.hypot(tx - ux, ty - uy)
    dist_to_to_node_m = np.hypot(tx - vx, ty - vy)
    arc_length_m = trip.arc_lengths[cand_arcs]

    # 3. Heading alignment features
    arc_b = trip.arc_bearings[cand_arcs]
    d_theta = inst_bearing[:, None] - arc_b
    heading_cos = np.cos(d_theta)
    heading_sin = np.sin(d_theta)
    heading_abs_diff_rad = np.abs(np.arctan2(heading_sin, heading_cos))

    sm_d_theta = sm_bearing[:, None] - arc_b
    smoothed_heading_cos = np.cos(sm_d_theta)

    # 4. Kinematic features broadcasted across K candidates
    gps_speed_mps = np.broadcast_to(speed[:, None], (n_obs, k))
    gps_accel_mps2 = np.broadcast_to(accel[:, None], (n_obs, k))
    gps_dt_sec = np.broadcast_to(dt[:, None], (n_obs, k))

    # 5. Topological node degree features
    from_node_in_deg = trip.node_in_degree[u_nodes].astype(np.float64)
    to_node_out_deg = trip.node_out_degree[v_nodes].astype(np.float64)

    # 6. Local temporal & topological continuity features (t-1 and t+1)
    # Check connectivity between candidates at t-1 and candidates at t
    prev_connected_min_dist = np.full((n_obs, k), 50.0, dtype=np.float64)
    next_connected_min_dist = np.full((n_obs, k), 50.0, dtype=np.float64)
    is_prev_top1_connected = np.zeros((n_obs, k), dtype=np.float64)
    is_next_top1_connected = np.zeros((n_obs, k), dtype=np.float64)

    if n_obs > 1:
        # Vectorized (N-1, K_prev, K_curr) connectivity check
        arcs_prev = cand_arcs[:-1, :, None]    # (N-1, K, 1)
        arcs_curr = cand_arcs[1:, None, :]     # (N-1, 1, K)
        v_prev = v_nodes[:-1, :, None]         # to_node of prev candidate
        u_curr = u_nodes[1:, None, :]          # from_node of curr candidate

        conn_prev_curr = (arcs_prev == arcs_curr) | (v_prev == u_curr)  # (N-1, K, K)

        dists_prev = np.broadcast_to(cand_dists[:-1, :, None], (n_obs - 1, k, k))
        masked_prev_dists = np.where(conn_prev_curr, dists_prev, 100.0)
        prev_connected_min_dist[1:] = np.min(masked_prev_dists, axis=1)
        is_prev_top1_connected[1:] = conn_prev_curr[:, 0, :].astype(np.float64)

        dists_curr_for_prev = np.broadcast_to(cand_dists[1:, None, :], (n_obs - 1, k, k))
        masked_next_dists = np.where(conn_prev_curr, dists_curr_for_prev, 100.0)
        next_connected_min_dist[:-1] = np.min(masked_next_dists, axis=2)
        is_next_top1_connected[:-1] = conn_prev_curr[:, :, 0].astype(np.float64)

        prev_connected_min_dist[0] = cand_dists[0]
        next_connected_min_dist[-1] = cand_dists[-1]
        is_prev_top1_connected[0] = 1.0
        is_next_top1_connected[-1] = 1.0

    # 7. 3-step rolling average perpendicular distance to the same candidate arc_id
    prev_tx = np.roll(trip.track_x, 1)
    prev_ty = np.roll(trip.track_y, 1)
    prev_tx[0], prev_ty[0] = trip.track_x[0], trip.track_y[0]

    next_tx = np.roll(trip.track_x, -1)
    next_ty = np.roll(trip.track_y, -1)
    next_tx[-1], next_ty[-1] = trip.track_x[-1], trip.track_y[-1]

    d_prev_same, _, _, _ = point_to_segments_distance(prev_tx[:, None], prev_ty[:, None], ux, uy, vx, vy)
    d_next_same, _, _, _ = point_to_segments_distance(next_tx[:, None], next_ty[:, None], ux, uy, vx, vy)
    window3_mean_perp_dist = (d_prev_same + cand_dists + d_next_same) / 3.0

    # Stack all 25 features in exact order of FEATURE_NAMES
    feat_stack = np.stack(
        [
            perp_dist_m,
            dist_rank,
            dist_ratio_to_min,
            dist_diff_to_min,
            dist_zscore_in_step,
            dist_to_second_best_gap,
            proj_alpha,
            endpoint_clamp_overshoot,
            dist_to_from_node_m,
            dist_to_to_node_m,
            arc_length_m,
            heading_cos,
            heading_sin,
            heading_abs_diff_rad,
            smoothed_heading_cos,
            gps_speed_mps,
            gps_accel_mps2,
            gps_dt_sec,
            from_node_in_deg,
            to_node_out_deg,
            prev_connected_min_dist,
            next_connected_min_dist,
            is_prev_top1_connected,
            is_next_top1_connected,
            window3_mean_perp_dist,
        ],
        axis=-1,
    )  # (N_obs, K, 25)

    X = feat_stack.reshape(n_obs * k, len(FEATURE_NAMES))
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    y = candidates.labels.reshape(n_obs * k) if candidates.labels is not None else None
    return X, y, candidates


def build_partition_dataset(
    trips: List[TripData],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, TripCandidates]]:
    """
    Build combined feature matrix X, target vector y, trip group array `groups`,
    and per-trip TripCandidates dictionary for a list of trips.
    """
    X_list = []
    y_list = []
    group_list = []
    cand_dict: Dict[str, TripCandidates] = {}

    for trip in trips:
        X_t, y_t, cands = extract_features_for_trip(trip)
        X_list.append(X_t)
        if y_t is not None:
            y_list.append(y_t)
        group_list.append(np.full(len(X_t), trip.trip_id))
        cand_dict[trip.trip_id] = cands

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list) if y_list else np.array([])
    groups_all = np.concatenate(group_list)
    return X_all, y_all, groups_all, cand_dict
