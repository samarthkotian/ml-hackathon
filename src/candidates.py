"""
Fast spatial candidate road-segment generation for PathSnap (Team SI No 107).
Uses multi-point segment indexing with scipy.spatial.cKDTree followed by
exact perpendicular point-to-segment distance ranking.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from src.config import TOP_K_CANDIDATES, MAX_CANDIDATE_RADIUS_M
from src.data_loader import TripData, point_to_segments_distance


@dataclass
class TripCandidates:
    trip_id: str
    cand_arcs: np.ndarray        # (N_obs, K) int32: candidate arc_ids
    cand_dists: np.ndarray       # (N_obs, K) float64: perpendicular distance (m)
    cand_alpha: np.ndarray       # (N_obs, K) float64: clamped projection ratio [0, 1]
    cand_raw_alpha: np.ndarray   # (N_obs, K) float64: unclamped projection ratio
    cand_proj_dist: np.ndarray   # (N_obs, K) float64: distance along arc from start node (m)
    labels: Optional[np.ndarray] = None  # (N_obs, K) int32: 1 if cand_arc == obs_ref_arc else 0
    candidate_miss_rate: Optional[float] = None


def generate_candidates_for_trip(
    trip: TripData,
    k: int = TOP_K_CANDIDATES,
    max_radius_m: float = MAX_CANDIDATE_RADIUS_M,
) -> TripCandidates:
    """
    Generate top-K candidate road segments for every GPS observation in `trip`.

    To ensure long road segments are not missed when querying by midpoint alone,
    we index 5 control points along every road segment (0%, 25%, 50%, 75%, 100%)
    in a 2D cKDTree, retrieve the candidate pool, deduplicate arc_ids per observation,
    and rank by exact perpendicular point-to-segment distance.
    """
    n_obs = len(trip.track)
    n_arcs = len(trip.arcs)
    k_eff = min(k, n_arcs)

    # Sample 5 points along each segment: alpha in {0.0, 0.25, 0.5, 0.75, 1.0}
    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    # Shape: (5, n_arcs) -> flatten to (5 * n_arcs, 2)
    pts_x = trip.arc_x1[None, :] + alphas[:, None] * (trip.arc_x2 - trip.arc_x1)[None, :]
    pts_y = trip.arc_y1[None, :] + alphas[:, None] * (trip.arc_y2 - trip.arc_y1)[None, :]
    sample_xy = np.column_stack([pts_x.ravel(), pts_y.ravel()])
    sample_arc_ids = np.tile(np.arange(n_arcs, dtype=np.int32), len(alphas))

    tree = cKDTree(sample_xy)
    # Query 6 * k_eff nearest control points to gather at least k_eff unique arcs
    q_k = min(len(sample_xy), max(k_eff * 6, 36))
    _, nn_idx = tree.query(np.column_stack([trip.track_x, trip.track_y]), k=q_k)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    raw_cand_arcs = sample_arc_ids[nn_idx]  # (n_obs, q_k)

    # Compute exact perpendicular distances for all q_k retrieved arcs vectorized
    cx1 = trip.arc_x1[raw_cand_arcs]
    cy1 = trip.arc_y1[raw_cand_arcs]
    cx2 = trip.arc_x2[raw_cand_arcs]
    cy2 = trip.arc_y2[raw_cand_arcs]

    dists, alpha, raw_alpha, proj_dist = point_to_segments_distance(
        trip.track_x[:, None],
        trip.track_y[:, None],
        cx1,
        cy1,
        cx2,
        cy2,
    )

    # Break exact perpendicular-distance ties between bidirectional twin arcs (u->v vs v->u)
    # using sub-millimeter (1e-4 m) heading alignment preference so the forward-directed twin
    # consistently wins rank 0 over the reverse-directed twin.
    cand_bearings = trip.arc_bearings[raw_cand_arcs]
    heading_tiebreak = 1e-4 * (
        1.0 - np.cos(trip.track_chord_bearing[:, None] - cand_bearings)
    )
    order = np.argsort(dists + heading_tiebreak, axis=1)
    sorted_arcs = np.take_along_axis(raw_cand_arcs, order, axis=1)
    sorted_dists = np.take_along_axis(dists, order, axis=1)
    sorted_alpha = np.take_along_axis(alpha, order, axis=1)
    sorted_raw_alpha = np.take_along_axis(raw_alpha, order, axis=1)
    sorted_proj_dist = np.take_along_axis(proj_dist, order, axis=1)

    cand_arcs = np.zeros((n_obs, k_eff), dtype=np.int32)
    cand_dists = np.full((n_obs, k_eff), np.inf, dtype=np.float64)
    cand_alpha = np.zeros((n_obs, k_eff), dtype=np.float64)
    cand_raw_alpha = np.zeros((n_obs, k_eff), dtype=np.float64)
    cand_proj_dist = np.zeros((n_obs, k_eff), dtype=np.float64)

    for i in range(n_obs):
        row_arcs = sorted_arcs[i]
        # Preserve first appearance of each unique arc_id in distance-sorted order
        _, unique_indices = np.unique(row_arcs, return_index=True)
        unique_indices.sort()
        sel = unique_indices[:k_eff]
        n_sel = len(sel)
        cand_arcs[i, :n_sel] = row_arcs[sel]
        cand_dists[i, :n_sel] = sorted_dists[i, sel]
        cand_alpha[i, :n_sel] = sorted_alpha[i, sel]
        cand_raw_alpha[i, :n_sel] = sorted_raw_alpha[i, sel]
        cand_proj_dist[i, :n_sel] = sorted_proj_dist[i, sel]
        if n_sel < k_eff:
            # Pad with last selected candidate if fewer than k_eff unique found
            cand_arcs[i, n_sel:] = cand_arcs[i, n_sel - 1]
            cand_dists[i, n_sel:] = cand_dists[i, n_sel - 1]
            cand_alpha[i, n_sel:] = cand_alpha[i, n_sel - 1]
            cand_raw_alpha[i, n_sel:] = cand_raw_alpha[i, n_sel - 1]
            cand_proj_dist[i, n_sel:] = cand_proj_dist[i, n_sel - 1]

    labels = None
    miss_rate = None
    if trip.obs_ref_arcs is not None and len(trip.obs_ref_arcs) == n_obs:
        labels = (cand_arcs == trip.obs_ref_arcs[:, None]).astype(np.int32)
        has_hit = np.any(labels == 1, axis=1)
        miss_rate = float(1.0 - np.mean(has_hit))

    return TripCandidates(
        trip_id=trip.trip_id,
        cand_arcs=cand_arcs,
        cand_dists=cand_dists,
        cand_alpha=cand_alpha,
        cand_raw_alpha=cand_raw_alpha,
        cand_proj_dist=cand_proj_dist,
        labels=labels,
        candidate_miss_rate=miss_rate,
    )
