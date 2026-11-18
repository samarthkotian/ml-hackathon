"""
Hidden Markov Model (HMM) and Log-Domain Viterbi Route Reconstruction
for PathSnap (Team SI No 107).

Models sequential GPS map-matching over a directed road network graph:
- Hidden states: candidate road segments S_t = {e_{t,0}, ..., e_{t,K-1}} at step t
- Emissions: calibrated classifier probability + spatial Gaussian likelihood
- Transitions: road-network graph connectivity (same segment, 1-hop connected,
  multi-hop shortest path for short links / GPS gaps, and disconnected penalty)
- Decoding: exact max-sum (log-domain) Viterbi dynamic programming with backpointers
  plus intermediate path bridging for complete connected route reconstruction.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data_loader import TripData
from src.candidates import TripCandidates


@dataclass
class HMMConfig:
    clf_weight: float = 1.6          # Weight on log P_clf(y=1 | x)
    dist_weight: float = 0.35        # Weight on spatial Gaussian emission -(d / sigma_e)^2
    heading_weight: float = 0.85     # Weight on spatial-chord directional alignment cos(d_theta)
    sigma_emission_m: float = 12.0   # Spatial GPS noise standard deviation (m)
    beta_transition_m: float = 18.0  # Exponential decay scale for |d_net - d_gps| (m)
    same_arc_bonus: float = 0.65     # Continuity prior for staying on the same road segment
    one_hop_bonus: float = 0.45      # Continuity prior for direct node-connected transition
    multi_hop_penalty: float = 1.15  # Per-hop penalty for 2..max_hops intermediate segments
    backward_penalty: float = 3.5    # Penalty for moving backward along the same directed arc
    uturn_penalty: float = 4.5       # Penalty for immediate U-turn flip onto reverse-twin arc
    disconnected_penalty: float = 9.5  # Penalty for topologically disconnected jump
    max_bfs_hops: int = 5            # Max intermediate graph hops searched


def _build_node_adjacency(trip: TripData) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Build outgoing adjacency list: u -> list of (v, arc_id, arc_length_m).
    """
    adj: Dict[int, List[Tuple[int, int, float]]] = {}
    u_arr = trip.arcs[:, 0]
    v_arr = trip.arcs[:, 1]
    lengths = trip.arc_lengths
    for arc_id in range(len(u_arr)):
        u = int(u_arr[arc_id])
        v = int(v_arr[arc_id])
        w = float(lengths[arc_id])
        if u not in adj:
            adj[u] = []
        adj[u].append((v, arc_id, w))
    return adj


def _precompute_node_reachability(
    trip: TripData, source_nodes: np.ndarray, max_hops: int = 5
) -> Dict[int, Dict[int, Tuple[int, float, List[int]]]]:
    """
    For each unique `to_node` of candidate segments, run bounded BFS/Dijkstra up to
    `max_hops` edges to find reachable `from_node`s of subsequent candidate segments.
    Returns:
      reach[src_node][dst_node] = (hops, path_length_m, intermediate_arc_ids)
    """
    adj = _build_node_adjacency(trip)
    reach: Dict[int, Dict[int, Tuple[int, float, List[int]]]] = {}

    for src in np.unique(source_nodes):
        src_int = int(src)
        visited: Dict[int, Tuple[int, float, List[int]]] = {src_int: (0, 0.0, [])}
        queue = deque([(src_int, 0, 0.0, [])])
        while queue:
            curr_u, hops, dist_so_far, path_arcs = queue.popleft()
            if hops >= max_hops:
                continue
            for nxt_v, arc_id, w in adj.get(curr_u, []):
                new_hops = hops + 1
                new_dist = dist_so_far + w
                if nxt_v not in visited or new_dist < visited[nxt_v][1]:
                    new_path = path_arcs + [arc_id]
                    visited[nxt_v] = (new_hops, new_dist, new_path)
                    queue.append((nxt_v, new_hops, new_dist, new_path))
        reach[src_int] = visited
    return reach


def viterbi_decode_trip(
    trip: TripData,
    candidates: TripCandidates,
    clf_probs: np.ndarray,
    config: Optional[HMMConfig] = None,
) -> Tuple[np.ndarray, List[int], Dict[str, float]]:
    """
    Run Log-Domain Viterbi decoding over a trip's candidate road segments.

    Parameters
    ----------
    trip : TripData
    candidates : TripCandidates
    clf_probs : shape (N_obs, K) or (N_obs * K,) float64
        Predicted probability P(y=1 | x) for each candidate road segment.
    config : HMMConfig

    Returns
    -------
    pred_obs_arcs : shape (N_obs,) int32
        Viterbi-decoded road segment arc_id for every GPS observation t.
    reconstructed_route : List[int]
        Chronological deduplicated sequence of traversed road segment arc_ids
        (including bridged intermediate segments for multi-hop transitions).
    stats : Dict[str, float]
        Diagnostic statistics (continuity, transition breakdown).
    """
    if config is None:
        config = HMMConfig()

    cand_arcs = candidates.cand_arcs
    cand_dists = candidates.cand_dists
    cand_proj = candidates.cand_proj_dist
    n_obs, k = cand_arcs.shape

    probs_2d = clf_probs.reshape(n_obs, k)
    clipped_probs = np.clip(probs_2d, 1e-5, 1.0 - 1e-5)

    # Normalize candidate probabilities within each step t so emissions are well-scaled
    row_sum = np.sum(clipped_probs, axis=1, keepdims=True)
    norm_probs = clipped_probs / np.maximum(row_sum, 1e-9)

    # Spatial-chord directional alignment cosine for each candidate
    cand_bearings = trip.arc_bearings[cand_arcs]
    cos_align = np.cos(trip.track_chord_bearing[:, None] - cand_bearings)

    # Emission log-probabilities: shape (N_obs, K)
    log_emit = (
        config.clf_weight * np.log(norm_probs)
        - config.dist_weight * 0.5 * (cand_dists / config.sigma_emission_m) ** 2
        + config.heading_weight * cos_align
    )

    # Precompute node reachability for all candidate endpoints in this trip
    u_nodes = trip.arcs[cand_arcs, 0]  # (N_obs, K) from_node_id
    v_nodes = trip.arcs[cand_arcs, 1]  # (N_obs, K) to_node_id
    arc_lens = trip.arc_lengths[cand_arcs]  # (N_obs, K)

    reach_map = _precompute_node_reachability(
        trip, v_nodes.ravel(), max_hops=config.max_bfs_hops
    )

    # GPS step distances and time deltas
    dx = np.diff(trip.track_x, prepend=trip.track_x[0])
    dy = np.diff(trip.track_y, prepend=trip.track_y[0])
    gps_step_dist = np.hypot(dx, dy)
    dt = np.diff(trip.track[:, 2], prepend=trip.track[0, 2] - 1.0)
    dt = np.maximum(dt, 1.0)

    # Viterbi DP tables
    dp = np.full((n_obs, k), -np.inf, dtype=np.float64)
    back = np.zeros((n_obs, k), dtype=np.int32)
    dp[0] = log_emit[0]

    for t in range(1, n_obs):
        d_gps = gps_step_dist[t]
        dt_t = dt[t]
        beta_eff = config.beta_transition_m * max(1.0, np.sqrt(dt_t))
        # Relax disconnected penalty during large GPS signal dropouts (dt > 10s)
        disc_pen = config.disconnected_penalty / (1.0 + 0.15 * max(0.0, dt_t - 2.0))

        # Construct (K_prev, K_curr) log-transition matrix for step t-1 -> t
        trans_mat = np.full((k, k), -disc_pen, dtype=np.float64)

        for i in range(k):
            a_prev = int(cand_arcs[t - 1, i])
            u_prev = int(u_nodes[t - 1, i])
            v_prev = int(v_nodes[t - 1, i])
            rem_prev = max(0.0, float(arc_lens[t - 1, i] - cand_proj[t - 1, i]))
            reach_from_v = reach_map.get(v_prev, {})

            for j in range(k):
                a_curr = int(cand_arcs[t, j])
                u_curr = int(u_nodes[t, j])
                v_curr = int(v_nodes[t, j])
                proj_curr = float(cand_proj[t, j])

                if a_prev == a_curr:
                    # Same road segment
                    delta_prog = proj_curr - float(cand_proj[t - 1, i])
                    if delta_prog < -0.6:
                        # Moving backward on a directed arc
                        trans_mat[i, j] = -config.backward_penalty - abs(delta_prog) / beta_eff
                    else:
                        detour = abs(max(0.0, delta_prog) - d_gps)
                        trans_mat[i, j] = config.same_arc_bonus - detour / beta_eff
                elif v_prev == u_curr:
                    # Directly connected road segment (1 hop)
                    d_net = rem_prev + proj_curr
                    detour = abs(d_net - d_gps)
                    score = config.one_hop_bonus - detour / beta_eff
                    if v_curr == u_prev:
                        # Immediate U-turn onto reverse-twin arc (u->v followed by v->u)
                        score -= config.uturn_penalty
                    trans_mat[i, j] = score
                elif u_curr in reach_from_v:
                    # Connected via 1..max_bfs_hops intermediate road segments
                    hops, mid_dist, _ = reach_from_v[u_curr]
                    d_net = rem_prev + mid_dist + proj_curr
                    detour = abs(d_net - d_gps)
                    trans_mat[i, j] = (
                        -config.multi_hop_penalty * hops - detour / beta_eff
                    )

        # Max-sum recursion: scores[i, j] = dp[t-1, i] + trans_mat[i, j]
        scores = dp[t - 1, :, None] + trans_mat
        best_prev = np.argmax(scores, axis=0)
        dp[t] = log_emit[t] + scores[best_prev, np.arange(k)]
        back[t] = best_prev

    # Backtrack optimal state sequence
    best_k_seq = np.zeros(n_obs, dtype=np.int32)
    best_k_seq[-1] = int(np.argmax(dp[-1]))
    for t in range(n_obs - 2, -1, -1):
        best_k_seq[t] = back[t + 1, best_k_seq[t + 1]]

    pred_obs_arcs = cand_arcs[np.arange(n_obs), best_k_seq]

    # Build chronological reconstructed route (bridging multi-hop intermediate arcs)
    reconstructed_route: List[int] = [int(pred_obs_arcs[0])]
    for t in range(1, n_obs):
        a_prev = int(pred_obs_arcs[t - 1])
        a_curr = int(pred_obs_arcs[t])
        if a_curr == a_prev:
            continue
        v_prev = int(trip.arcs[a_prev, 1])
        u_curr = int(trip.arcs[a_curr, 0])
        if v_prev != u_curr:
            reach_from_v = reach_map.get(v_prev, {})
            if u_curr in reach_from_v:
                _, _, mid_arcs = reach_from_v[u_curr]
                for ma in mid_arcs:
                    if not reconstructed_route or reconstructed_route[-1] != ma:
                        reconstructed_route.append(int(ma))
        if not reconstructed_route or reconstructed_route[-1] != a_curr:
            reconstructed_route.append(a_curr)

    continuity = compute_route_continuity(trip, pred_obs_arcs, reach_map=reach_map)
    stats = {
        "route_continuity": continuity["connected_continuity"],
        "strict_1hop_continuity": continuity["strict_1hop_continuity"],
        "n_reconstructed_segments": float(len(reconstructed_route)),
    }
    return pred_obs_arcs, reconstructed_route, stats


def compute_route_continuity(
    trip: TripData,
    pred_obs_arcs: np.ndarray,
    reach_map: Optional[Dict[int, Dict[int, Tuple[int, float, List[int]]]]] = None,
) -> Dict[str, float]:
    """
    Compute topological route continuity of consecutive predicted road segments:
    - `strict_1hop_continuity`: fraction of steps (t-1, t) where pred_arc[t] == pred_arc[t-1]
      or to_node(pred_arc[t-1]) == from_node(pred_arc[t]).
    - `connected_continuity`: fraction of steps where pred_arc[t] is topologically connected
      from pred_arc[t-1] in the road network (same arc, direct 1-hop, or connected short link).
    """
    n = len(pred_obs_arcs)
    if n <= 1:
        return {"strict_1hop_continuity": 1.0, "connected_continuity": 1.0}

    if reach_map is None:
        v_all = trip.arcs[np.unique(pred_obs_arcs), 1]
        reach_map = _precompute_node_reachability(trip, v_all, max_hops=4)

    strict_ok = 0
    conn_ok = 0
    for t in range(1, n):
        a1 = int(pred_obs_arcs[t - 1])
        a2 = int(pred_obs_arcs[t])
        if a1 == a2:
            strict_ok += 1
            conn_ok += 1
        else:
            v1 = int(trip.arcs[a1, 1])
            u2 = int(trip.arcs[a2, 0])
            if v1 == u2:
                strict_ok += 1
                conn_ok += 1
            elif u2 in reach_map.get(v1, {}):
                conn_ok += 1

    return {
        "strict_1hop_continuity": float(strict_ok / (n - 1)),
        "connected_continuity": float(conn_ok / (n - 1)),
    }


def compute_sequence_route_metrics(
    reconstructed_route: List[int], reference_route: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Compare the reconstructed sequence of road segments against the ground-truth
    .route sequence using segment-set Precision, Recall, F1, and Jaccard similarity.
    """
    if reference_route is None or len(reference_route) == 0:
        return {
            "route_seq_precision": 0.0,
            "route_seq_recall": 0.0,
            "route_seq_f1": 0.0,
            "route_jaccard": 0.0,
        }

    pred_set = set(int(x) for x in reconstructed_route)
    ref_set = set(int(x) for x in reference_route)
    inter = len(pred_set & ref_set)
    union = len(pred_set | ref_set)

    prec = inter / max(1, len(pred_set))
    rec = inter / max(1, len(ref_set))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    jaccard = inter / max(1, union)

    return {
        "route_seq_precision": float(prec),
        "route_seq_recall": float(rec),
        "route_seq_f1": float(f1),
        "route_jaccard": float(jaccard),
    }
