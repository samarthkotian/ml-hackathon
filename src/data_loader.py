"""
Data loading, coordinate projection, road-network graph construction,
and observation-to-reference-route alignment for PathSnap (Team SI No 107).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.paths import safe_child, validate_trip_id

from src.config import (
    EARTH_RADIUS_M,
    TRAIN_DIR,
    STUDENT_TEST_DIR,
    EVALUATOR_TEST_DIR,
)


def project_lonlat_to_meters(
    lon: np.ndarray, lat: np.ndarray, ref_lon: float, ref_lat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project angular WGS84 (longitude, latitude) in degrees to a local tangent-plane
    metric coordinate system (East x, North y in meters) centered at (ref_lon, ref_lat).

    Justification:
    Ordinary Euclidean distance in degrees distorts East-West vs North-South distances
    by a factor of cos(latitude). At latitude ~50 deg, 1 deg longitude is ~71.7 km whereas
    1 deg latitude is ~111.2 km (a 35% distortion). Local equirectangular / tangent-plane
    projection centered on each trip's centroid achieves <0.01% metric error over city-scale
    trajectories (5-30 km extent).
    """
    cos_lat = np.cos(np.radians(ref_lat))
    x = np.radians(lon - ref_lon) * EARTH_RADIUS_M * cos_lat
    y = np.radians(lat - ref_lat) * EARTH_RADIUS_M
    return x, y


def point_to_segments_distance(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized perpendicular/clamped point-to-segment distance in meters.

    Parameters
    ----------
    px, py : shape (N,) or (N, K)
    x1, y1, x2, y2 : shape (M,) or (N, K)

    Returns
    -------
    dist : perpendicular distance to segment (meters)
    alpha : clamped projection fraction in [0, 1] along segment
    raw_alpha : unclamped projection fraction along infinite line
    proj_dist_along : distance in meters from segment start to projected point
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1e-12)

    raw_alpha = ((px - x1) * dx + (py - y1) * dy) / safe_len_sq
    alpha = np.clip(raw_alpha, 0.0, 1.0)
    proj_x = x1 + alpha * dx
    proj_y = y1 + alpha * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    proj_dist_along = alpha * np.sqrt(seg_len_sq)
    return dist, alpha, raw_alpha, proj_dist_along


def align_track_to_route(
    tx: np.ndarray,
    ty: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align each GPS observation t in [0..N-1] with its corresponding ground-truth
    reference road segment arc_id from the chronological .route sequence.

    Uses monotonic dynamic programming along the ordered .route sequence so that
    even if a trajectory loops or crosses itself, each timestamp t is matched to
    the chronologically consistent pass along .route.

    Returns
    -------
    obs_ref_arcs : shape (N,) int array of ground-truth arc_id per GPS point
    obs_ref_dists : shape (N,) float array of distance (m) to the assigned reference arc
    """
    if route is None or len(route) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    r_from = arcs[route, 0]
    r_to = arcs[route, 1]
    rx1, ry1 = nx[r_from], ny[r_from]
    rx2, ry2 = nx[r_to], ny[r_to]

    # Compute (N, M) distance matrix between N GPS points and M route segments
    dist_mat, _, _, _ = point_to_segments_distance(
        tx[:, None],
        ty[:, None],
        rx1[None, :],
        ry1[None, :],
        rx2[None, :],
        ry2[None, :],
    )

    N, M = dist_mat.shape
    if M == 1:
        return np.full(N, route[0], dtype=int), dist_mat[:, 0]

    # Fast vectorized monotonic dynamic programming allowing forward progression
    # along route index m (and small step-backs of 1 segment for noisy GPS jitter at junctions)
    dp = np.full((N, M), np.inf, dtype=np.float64)
    back = np.zeros((N, M), dtype=np.int32)

    # Initialize near start of route (with gentle penalty for starting far into the route)
    dp[0] = dist_mat[0] + np.arange(M) * 2.0

    # Max forward jump in route segments per GPS observation (handles short segments or time gaps)
    max_jump = min(M, 12)

    for i in range(1, N):
        # Candidate transitions: m_prev in [m - max_jump .. m + 1]
        best_prev_cost = dp[i - 1].copy()
        best_prev_idx = np.arange(M, dtype=np.int32)

        # Allow staying on same segment m (cost 0)
        # Allow advancing 1..max_jump segments forward
        for step in range(1, max_jump + 1):
            # Transition penalty for skipping many segments in a single step
            step_pen = 0.5 * max(0, step - 2)
            shifted_cost = np.empty(M, dtype=np.float64)
            shifted_cost[:step] = np.inf
            shifted_cost[step:] = dp[i - 1, :-step] + step_pen
            mask = shifted_cost < best_prev_cost
            best_prev_cost[mask] = shifted_cost[mask]
            best_prev_idx[mask] = np.arange(M, dtype=np.int32)[mask] - step

        # Allow 1-segment step back at noisy intersections with small penalty
        back1_cost = np.empty(M, dtype=np.float64)
        back1_cost[-1] = np.inf
        back1_cost[:-1] = dp[i - 1, 1:] + 3.0
        mask_b = back1_cost < best_prev_cost
        best_prev_cost[mask_b] = back1_cost[mask_b]
        best_prev_idx[mask_b] = np.arange(M, dtype=np.int32)[mask_b] + 1

        dp[i] = dist_mat[i] + best_prev_cost
        back[i] = best_prev_idx

    mono_idx = np.zeros(N, dtype=np.int32)
    # Slight bias toward ending near the end of route
    end_costs = dp[-1] + np.arange(M - 1, -1, -1) * 1.5
    mono_idx[-1] = int(np.argmin(end_costs))
    for i in range(N - 2, -1, -1):
        mono_idx[i] = back[i + 1, mono_idx[i + 1]]

    obs_ref_arcs = route[mono_idx].astype(int)
    obs_ref_dists = dist_mat[np.arange(N), mono_idx]
    return obs_ref_arcs, obs_ref_dists


def compute_spatial_chord_bearing(
    x: np.ndarray, y: np.ndarray, min_disp_m: float = 18.0
) -> np.ndarray:
    """
    Compute stationary-robust trajectory travel bearing at every GPS observation t
    by finding the nearest preceding and succeeding GPS points at least `min_disp_m`
    meters away. Prevents stationary GPS jitter at red lights or slow traffic from
    corrupting vehicle heading.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    fwd_dx = np.zeros(n, dtype=np.float64)
    fwd_dy = np.zeros(n, dtype=np.float64)
    bwd_dx = np.zeros(n, dtype=np.float64)
    bwd_dy = np.zeros(n, dtype=np.float64)

    r_ptr = 0
    for i in range(n):
        while r_ptr < n - 1 and np.hypot(x[r_ptr] - x[i], y[r_ptr] - y[i]) < min_disp_m:
            r_ptr += 1
        fwd_dx[i] = x[r_ptr] - x[i]
        fwd_dy[i] = y[r_ptr] - y[i]

    l_ptr = n - 1
    for i in range(n - 1, -1, -1):
        while l_ptr > 0 and np.hypot(x[i] - x[l_ptr], y[i] - y[l_ptr]) < min_disp_m:
            l_ptr -= 1
        bwd_dx[i] = x[i] - x[l_ptr]
        bwd_dy[i] = y[i] - y[l_ptr]

    f_norm = np.hypot(fwd_dx, fwd_dy) + 1e-6
    b_norm = np.hypot(bwd_dx, bwd_dy) + 1e-6
    vx = fwd_dx / f_norm + bwd_dx / b_norm
    vy = fwd_dy / f_norm + bwd_dy / b_norm
    return np.arctan2(vy, vx)


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: np.ndarray          # (N_obs, 3): lon, lat, timestamp
    nodes: np.ndarray          # (N_nodes, 2): lon, lat
    arcs: np.ndarray           # (N_arcs, 2): from_node_id, to_node_id
    route: Optional[np.ndarray] = None  # (N_route,): chronological arc_ids

    # Derived metric coordinates & road segment geometry
    ref_lon: float = 0.0
    ref_lat: float = 0.0
    track_x: np.ndarray = field(default_factory=lambda: np.array([]))
    track_y: np.ndarray = field(default_factory=lambda: np.array([]))
    track_chord_bearing: np.ndarray = field(default_factory=lambda: np.array([]))
    nodes_x: np.ndarray = field(default_factory=lambda: np.array([]))
    nodes_y: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_x1: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_y1: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_x2: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_y2: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_mid_x: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_mid_y: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_bearings: np.ndarray = field(default_factory=lambda: np.array([]))
    node_out_degree: np.ndarray = field(default_factory=lambda: np.array([]))
    node_in_degree: np.ndarray = field(default_factory=lambda: np.array([]))

    # Observation-level ground truth aligned from .route
    obs_ref_arcs: Optional[np.ndarray] = None
    obs_ref_dists: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.ref_lon = float(np.mean(self.track[:, 0]))
        self.ref_lat = float(np.mean(self.track[:, 1]))

        self.track_x, self.track_y = project_lonlat_to_meters(
            self.track[:, 0], self.track[:, 1], self.ref_lon, self.ref_lat
        )
        self.track_chord_bearing = compute_spatial_chord_bearing(
            self.track_x, self.track_y, min_disp_m=18.0
        )
        self.nodes_x, self.nodes_y = project_lonlat_to_meters(
            self.nodes[:, 0], self.nodes[:, 1], self.ref_lon, self.ref_lat
        )

        u = self.arcs[:, 0]
        v = self.arcs[:, 1]
        self.arc_x1 = self.nodes_x[u]
        self.arc_y1 = self.nodes_y[u]
        self.arc_x2 = self.nodes_x[v]
        self.arc_y2 = self.nodes_y[v]
        self.arc_mid_x = 0.5 * (self.arc_x1 + self.arc_x2)
        self.arc_mid_y = 0.5 * (self.arc_y1 + self.arc_y2)

        dx = self.arc_x2 - self.arc_x1
        dy = self.arc_y2 - self.arc_y1
        self.arc_lengths = np.hypot(dx, dy)
        self.arc_bearings = np.arctan2(dy, dx)  # radians in [-pi, pi]

        n_nodes = len(self.nodes)
        self.node_out_degree = np.bincount(u, minlength=n_nodes)
        self.node_in_degree = np.bincount(v, minlength=n_nodes)

        if self.route is not None and len(self.route) > 0:
            self.obs_ref_arcs, self.obs_ref_dists = align_track_to_route(
                self.track_x,
                self.track_y,
                self.nodes_x,
                self.nodes_y,
                self.arcs,
                self.route,
            )


def load_trip_from_dir(trip_dir: str, trip_id: str, partition: str = "unknown") -> TripData:
    """Load a single trip (.track, .nodes, .arcs, and optional .route) from disk."""
    validate_trip_id(trip_id)
    track_path = safe_child(trip_dir, f"{trip_id}.track")
    nodes_path = safe_child(trip_dir, f"{trip_id}.nodes")
    arcs_path = safe_child(trip_dir, f"{trip_id}.arcs")
    route_path = safe_child(trip_dir, f"{trip_id}.route")

    track = np.loadtxt(track_path, dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)

    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)

    arcs = np.loadtxt(arcs_path, dtype=np.int32)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route = None
    if os.path.exists(route_path):
        r_raw = np.loadtxt(route_path, dtype=np.int32)
        route = r_raw.reshape(-1) if r_raw.size > 0 else np.array([], dtype=np.int32)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def load_partition(partition_name: str, base_dir: Optional[str] = None) -> List[TripData]:
    """Load all trips in a given partition ('train', 'student_test', 'evaluator_test')."""
    if base_dir is None:
        if partition_name == "train":
            base_dir = TRAIN_DIR
        elif partition_name == "student_test":
            base_dir = STUDENT_TEST_DIR
        elif partition_name == "evaluator_test":
            base_dir = EVALUATOR_TEST_DIR
        else:
            raise ValueError(f"Unknown partition: {partition_name}")

    if not os.path.exists(base_dir):
        return []

    trip_ids = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    trips = []
    for tid in trip_ids:
        tdir = os.path.join(base_dir, tid)
        if os.path.exists(os.path.join(tdir, f"{tid}.track")):
            trips.append(load_trip_from_dir(tdir, tid, partition=partition_name))
    return trips
