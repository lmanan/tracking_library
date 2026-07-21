"""infer_graph_cut.py — APT-style graph-cut tracklet stitching, formulated as an exact ILP (gurobipy).

A third sibling to infer_node_edge_selection.py (motile min-cost-flow selection) and
infer_clusters.py (hierarchical clustering). This one reproduces APT's *labeling* graph cut
(deepnet/link_trajectories.py: group_graph_cut / get_id_cluster_centers / get_graph_cut_unary
/ get_graph_cut_pairwise / k_way_graph_cut) but replaces alpha-expansion with an exact ILP.

Why an ILP: APT's graph cut adds a temporal-overlap "repulsion" via NEGATIVE pairwise weights,
which makes the energy non-submodular; the gco alpha-expansion solver then aborts (SIGABRT,
"add_term2: B + C >= 0"). An ILP has no submodularity requirement -- overlap is a native linear
term -- so it never has that failure mode and returns a global optimum.

Temporal overlap is weighted by `overlap_penalty` (p), DEFAULT 1.0:
  p > 0  (default)  SOFT: pay p for each overlapping pair that shares a non-outlier label. This
                    is the linear analogue of APT's repulsion, and strong appearance evidence can
                    override a spurious overlap. Best on every cohort7 movie.
  p = 0  (opt in)   HARD: must-not-link constraint x[i,l] + x[j,l] <= 1. Cannot be overridden, so
                    a tracklet overlapping all K identities is forced to the outlier label.

Two modes (config `mode`):
  apt_faithful           — prototypes from hierarchical clustering of the clean, long (>50f)
                           subset -> per-tracklet UNARY (dist to nearest prototype) + Potts
                           MOTION pairwise (attraction between temporally-local, motion-
                           consistent tracklets) + temporal-overlap repulsion + K identities.
  correlation_clustering — no prototypes: pairwise id-embedding ATTRACTION (k-NN) + overlap
                           repulsion + K identities (minimum-cost-multicut style).

Faithfulness notes:
  * EVERY input tracklet is labeled. In APT the graph cut labels all pure tracklets; the >=6
    frame cutoff is only APT's embedding-eligibility, and the >50 cutoff only selects prototype
    centres. (Our input CSV is already the get_pure_tracklets >6 set.)
  * Occlusion: APT weights several terms by pTrkTag occlusion tags. The tracking_library CSV has
    no occlusion column, so occ=0 -> APT's occlusion terms vanish cleanly (unary unchanged,
    centre weighting uniform, occ<0.2 filter passes all).
  * Motion pairwise here is a faithful-in-spirit reimplementation of get_motion_links +
    get_graph_cut_pairwise from the CSV keypoints (endpoint centroid + single-step velocity,
    gap-penalised cost, top-k symmetric, a = max(0, 4 - m/len_div)). d_norm (APT's maxcost*2)
    is approximated by 2x the median per-frame centroid step.

Same solver contract as infer_clusters.py: single positional config arg; honors
pinned_nodes_false (drop -> excluded), pinned_nodes_true (keep -> non-outlier),
pinned_chains (pin edge -> must-link / same identity); writes via save_solution_keypoints_csv.

Usage:  python infer_graph_cut.py <configs_graph_cut.yaml>
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import gurobipy as gp
from gurobipy import GRB

from x_utils.general_utils import load_config, print_config
from data_utils.save_tracklet_csv_data import save_solution_keypoints_csv
# Shared appearance scaffolding (drop = slice, pin = contract) from the clustering sibling.
from infer_clusters import (
    load_tracklets, sample_per_tracklet, id_distance_matrix, mustlink_groups,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Geometry (for the motion pairwise): per-tracklet endpoints + single-step velocity.
# --------------------------------------------------------------------------------------
def load_geometry(csv_path, num_keypoints, t_min, t_max):
    """{tid: dict(t_start, t_end, length, first_c, last_c, first_v, last_v)} where *_c are
    (y, x) centroids (mean over keypoints) of the first/last detection and *_v the one-step
    centroid velocity at each end."""
    ycols = [f"kp{i}_y" for i in range(num_keypoints)]
    xcols = [f"kp{i}_x" for i in range(num_keypoints)]
    df = pd.read_csv(csv_path, sep=r"\s+", usecols=["t", "tracklet_id"] + ycols + xcols)
    if t_min is not None:
        df = df[df["t"] >= t_min]
    if t_max is not None:
        df = df[df["t"] <= t_max]
    cy = df[ycols].to_numpy(np.float64).mean(axis=1)
    cx = df[xcols].to_numpy(np.float64).mean(axis=1)
    df = df.assign(_cy=cy, _cx=cx)
    geom = {}
    for tid, sub in df.groupby("tracklet_id"):
        sub = sub.sort_values("t")
        c = sub[["_cy", "_cx"]].to_numpy()
        t = sub["t"].to_numpy()
        fv = (c[1] - c[0]) if len(c) > 1 else np.zeros(2)
        lv = (c[-1] - c[-2]) if len(c) > 1 else np.zeros(2)
        geom[int(tid)] = dict(t_start=int(t.min()), t_end=int(t.max()), length=len(c),
                              first_c=c[0], last_c=c[-1], first_v=fv, last_v=lv)
    return geom


# --------------------------------------------------------------------------------------
# Super-nodes: apply drop (remove) + pin (contract must-link groups). One entry per node.
# --------------------------------------------------------------------------------------
def build_super_nodes(tids, E, frames, geom, dropped, kept, pinned_chains, n_ex, rng):
    """Returns list of super-node dicts: members(tid list), embs[n_ex,D] (pooled+resampled),
    t_start, t_end, length, first_c/last_c/first_v/last_v (from earliest/latest member), kept."""
    pos = {t: i for i, t in enumerate(tids)}
    work = [t for t in tids if t not in dropped]
    groups = [sorted(g) for g in mustlink_groups(pinned_chains, set(work))]

    nodes = []
    for g in groups:
        embs_all = np.concatenate([E[pos[t]] for t in g], axis=0)
        sel = rng.choice(len(embs_all), size=n_ex, replace=(len(embs_all) < n_ex))
        fr = np.unique(np.concatenate([frames[pos[t]] for t in g]))
        gg = [t for t in g if t in geom]
        first = min(gg, key=lambda t: geom[t]["t_start"]) if gg else g[0]
        last = max(gg, key=lambda t: geom[t]["t_end"]) if gg else g[0]
        nodes.append(dict(
            members=g, embs=embs_all[sel], frames=fr,
            t_start=int(geom[first]["t_start"]) if gg else int(fr.min()),
            t_end=int(geom[last]["t_end"]) if gg else int(fr.max()),
            length=int(len(fr)),
            first_c=geom[first]["first_c"] if gg else np.zeros(2),
            last_c=geom[last]["last_c"] if gg else np.zeros(2),
            first_v=geom[first]["first_v"] if gg else np.zeros(2),
            last_v=geom[last]["last_v"] if gg else np.zeros(2),
            kept=any(t in kept for t in g)))
    return nodes


def intra_spread(embs_stack):
    """dist_diag[i] = median pairwise L2 among tracklet i's own samples (APT get_id_dist_mat_diag)."""
    out = np.zeros(len(embs_stack))
    for i, e in enumerate(embs_stack):
        d = np.linalg.norm(e[:, None, :] - e[None, :, :], axis=-1)
        iu = np.triu_indices(len(e), k=1)
        out[i] = np.median(d[iu]) if len(iu[0]) else 0.0
    return out


def spatial_overlap(csv_path, num_keypoints, t_min, t_max):
    """{tid: mean max-bbox-overlap with any other animal in the same frame} — APT's spatial `oa`
    (get_overlap_value / _frame_overlaps): per detection, max over co-present detections of
    intersection/min_area on 10%-padded keypoint bboxes; averaged over the tracklet's frames.
    NB this is SPATIAL overlap (animals usually separated -> small), not temporal co-occupancy
    (animals always coexist -> ~1) which would reject every tracklet from the clean subset."""
    ycols = [f"kp{i}_y" for i in range(num_keypoints)]
    xcols = [f"kp{i}_x" for i in range(num_keypoints)]
    df = pd.read_csv(csv_path, sep=r"\s+", usecols=["t", "tracklet_id"] + ycols + xcols)
    if t_min is not None:
        df = df[df["t"] >= t_min]
    if t_max is not None:
        df = df[df["t"] <= t_max]
    Y, X = df[ycols].to_numpy(float), df[xcols].to_numpy(float)
    # nanmin/nanmax over keypoints, exactly as APT _frame_overlaps (a missing keypoint still
    # yields a valid box from the remaining ones).
    y0, y1 = np.nanmin(Y, axis=1), np.nanmax(Y, axis=1)
    x0, x1 = np.nanmin(X, axis=1), np.nanmax(X, axis=1)
    yl, xl = (y1 - y0) * 0.1, (x1 - x0) * 0.1
    df = df.assign(_y0=y0 - yl, _y1=y1 + yl, _x0=x0 - xl, _x1=x1 + xl)

    osum, ocnt = defaultdict(float), defaultdict(int)
    for _, sub in df.groupby("t"):
        b = sub[["_y0", "_y1", "_x0", "_x1"]].to_numpy()
        tid = sub["tracklet_id"].to_numpy()
        m = len(b)
        if m == 1:
            osum[int(tid[0])] += 0.0; ocnt[int(tid[0])] += 1
            continue
        iy0 = np.maximum(b[:, 0:1], b[:, 0]); iy1 = np.minimum(b[:, 1:2], b[:, 1])
        ix0 = np.maximum(b[:, 2:3], b[:, 2]); ix1 = np.minimum(b[:, 3:4], b[:, 3])
        inter = np.maximum(0.0, iy1 - iy0) * np.maximum(0.0, ix1 - ix0)
        area = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])
        min_area = np.minimum(area[:, None], area[None, :])
        ov = np.where(min_area > 0, inter / min_area, 0.0)
        np.fill_diagonal(ov, 0.0)
        mx = ov.max(1)
        for k in range(m):
            osum[int(tid[k])] += mx[k]; ocnt[int(tid[k])] += 1
    return {t: osum[t] / ocnt[t] for t in osum if ocnt[t] > 0}


# --------------------------------------------------------------------------------------
# apt_faithful: prototypes (get_id_cluster_centers) + unary (get_graph_cut_unary)
# --------------------------------------------------------------------------------------
def get_prototypes(nodes, dist_diag, oa, num_tracks, cfg):
    """Hierarchical clustering of the clean, long subset -> occupancy/length-weighted mean
    prototype centres. Mirrors get_id_cluster_centers (occ term inactive: occ=0)."""
    E_mean = np.stack([nd["embs"].mean(axis=0) for nd in nodes])   # [n, D]
    lengths = np.array([nd["length"] for nd in nodes], float)
    close_thresh = max(0.5, np.percentile(dist_diag, 100 - cfg["close_thresh_percentile"]))

    clean = np.where((lengths > cfg["centre_min_len"]) & (oa < cfg["centre_max_overlap"]) &
                     (dist_diag < close_thresh))[0]
    if len(clean) < (num_tracks or 2):     # too strict -> relax to all nodes
        logger.warning("Only %d clean tracklets for prototypes; relaxing filter to all.", len(clean))
        clean = np.arange(len(nodes))

    D = id_distance_matrix(np.stack([nodes[i]["embs"] for i in clean]))
    if len(clean) == 1:
        F = np.array([1])
    else:
        Z = linkage(squareform(D, checks=False), "average")
        if num_tracks:
            F = fcluster(Z, t=int(num_tracks), criterion="maxclust")
        else:
            F = fcluster(Z, t=cfg["cluster_threshold"], criterion="distance")

    total_len = lengths.sum()

    # Effective min-cluster fraction. In threshold mode, if `num_animals` is set and the
    # configured min_cluster_frac admits FEWER than num_animals prototypes, lower the fraction
    # just enough to admit at least num_animals clusters -- i.e. down to the size (frac of total
    # length) of the num_animals-th largest cluster. If the configured fraction already yields
    # >= num_animals prototypes, it is left untouched. Only lowers, never raises, and only in
    # threshold mode (num_tracks unset); known mode skips the fraction filter entirely.
    frac = cfg["min_cluster_frac"]
    num_animals = cfg.get("num_animals")
    if not num_tracks:
        sizes = sorted((lengths[clean[F == lab]].sum() / total_len
                        for lab in range(1, int(F.max()) + 1)), reverse=True)
        if num_animals and sum(s >= frac for s in sizes) < int(num_animals):
            target = int(num_animals)
            frac = sizes[target - 1] if len(sizes) >= target else 0.0
        short = num_animals and len(sizes) < int(num_animals)
        logger.info("threshold mode: operating min_cluster_frac=%.4f (configured %.3f, "
                    "num_animals=%s) -> %d prototype(s) from %d raw cluster(s)%s.",
                    frac, cfg["min_cluster_frac"], num_animals,
                    sum(s >= frac for s in sizes), len(sizes),
                    f" -- only {len(sizes)} raw clusters at cluster_threshold="
                    f"{cfg['cluster_threshold']}; lower it for more" if short else "")

    centres = []
    for lab in range(1, F.max() + 1):
        members = clean[F == lab]
        cl = lengths[members].sum()
        if (not num_tracks) and cl / total_len < frac:
            continue                        # drop tiny clusters (threshold mode only)
        w = lengths[members]
        centres.append(np.average(E_mean[members], axis=0, weights=w))
    return np.stack(centres) if centres else E_mean[[int(np.argmax(lengths))]]


def get_unary(nodes, centres, cfg):
    """u[i, l] for l in 0..K. u[i,0] = outlier_unary; u[i,k] = mean sample-to-centre distance."""
    n, K = len(nodes), len(centres)
    U = np.zeros((n, K + 1))
    U[:, 0] = cfg["outlier_unary"]
    for i, nd in enumerate(nodes):
        e = nd["embs"]                                  # [n_ex, D]
        for k, c in enumerate(centres):
            U[i, k + 1] = np.linalg.norm(e - c[None, :], axis=1).mean()
    return U


def get_motion_pairwise(nodes, cfg):
    """Attraction a_ij >= 0 between temporally-local, motion-consistent tracklets. Faithful-in-
    spirit to get_motion_links (overlap==0 branch) + get_graph_cut_pairwise."""
    n = len(nodes)
    # d_norm ~ APT maxcost*2, approximated by 2x median per-frame centroid step.
    steps = [np.linalg.norm(nd["last_v"]) for nd in nodes if nd["length"] > 1]
    d_norm = 2.0 * (np.median(steps) if steps else 1.0)
    d_norm = max(d_norm, 1e-6)
    max_gap = cfg["motion_max_fr_gap"]

    cand = defaultdict(list)                            # i -> list of (a_ij, j)
    for i in range(n):
        ni = nodes[i]
        for j in range(n):
            if i == j:
                continue
            nj = nodes[j]
            if ni["t_end"] < nj["t_start"]:            # i before j
                g = nj["t_start"] - ni["t_end"]
                if g <= 0 or g > max_gap:
                    continue
                pred_i = ni["last_c"] + ni["last_v"] * (g / 2.0)
                pred_j = nj["first_c"] - nj["first_v"] * (g / 2.0)
                spatial = np.linalg.norm(pred_i - pred_j)
                m = spatial / d_norm / g + (g - 1) * cfg["motion_gap_penalty"]
                len_div = min(1.0, min(ni["length"], nj["length"]) / 3.0)
                a = max(0.0, cfg["pairwise_attract_max"] - m / max(len_div, 1e-6))
                if a > 0:
                    cand[i].append((a, j))
                    cand[j].append((a, i))     # bidirectional: j's motion links include its
                    #                            predecessor i, else the mutual top-k below (which
                    #                            needs i in topk[j]) can never hold and drops every edge.
    # top-k per node, then keep only mutual (both rank the other in their top-k) edges
    topk = {i: set(j for _, j in sorted(v, reverse=True)[: cfg["pairwise_top_k"]])
            for i, v in cand.items()}
    edges = {}
    for i in cand:
        for a, j in cand[i]:
            if j in topk.get(i, ()) and i in topk.get(j, ()):
                key = (min(i, j), max(i, j))
                edges[key] = max(edges.get(key, 0.0), a)
    return edges


# --------------------------------------------------------------------------------------
# correlation_clustering: k-NN id-embedding attraction
# --------------------------------------------------------------------------------------
def get_correlation_edges(D, cfg):
    """Attraction a_ij = margin - D_ij for each node's k nearest id-neighbours with D < margin."""
    margin = cfg["corr_margin"] if cfg.get("corr_margin") is not None else cfg["cluster_threshold"]
    n = D.shape[0]
    k = cfg["corr_knn"]
    edges = {}
    for i in range(n):
        order = np.argsort(D[i])
        cnt = 0
        for j in order:
            if j == i:
                continue
            if D[i, j] >= margin or cnt >= k:
                break
            key = (min(i, j), max(i, j))
            edges[key] = max(edges.get(key, 0.0), float(margin - D[i, j]))
            cnt += 1
    return edges


def get_overlap_pairs(nodes, cfg):
    """Pairs that co-occupy > overlap_frac_thresh of the shorter node's frames. These get the
    repulsion treatment in the ILP: a soft penalty p (default) or a hard must-not-link (p = 0).

    Overlap is measured on the frames a node ACTUALLY occupies, not on its
    [t_start, t_end] span. The two differ for a *contracted* super-node: a pin across a
    time gap (build_super_nodes must-links the pair) produces a node whose span covers the
    whole gap while it occupies only its members' frames. Measured by span, such a node
    "overlaps" nearly every other node in the movie and collects a repulsion penalty
    against each -- so the ILP dumps it into the outlier class and the pin is effectively
    self-defeating. (Observed: one pin across a ~9.6k-frame gap produced 522 span-overlap
    pairs where only 23 were real.)

    The span test is kept as a cheap pre-filter: disjoint spans cannot share frames.
    """
    n = len(nodes)
    # overlap_weight_mode scales the per-pair repulsion by HOW MUCH two tracklets overlap
    # (APT's repulsion is length/overlap-scaled `-min(min_len, ovv*min_len)`, not flat):
    #   constant -> 1.0            (original: flat penalty per overlapping pair)
    #   frames   -> shared frames  (a 20-frame overlap repels 20x a 1-frame overlap)
    #   frac     -> shared frames / shorter length  (in [0,1])
    #   apt      -> min(shorter length, shared frames)  (APT's own form)
    mode = cfg.get("overlap_weight_mode", "constant")
    pairs = []
    for i in range(n):
        fi, li = nodes[i]["frames"], nodes[i]["length"]
        for j in range(i + 1, n):
            # cheap reject: spans don't even touch -> no shared frames
            if min(nodes[i]["t_end"], nodes[j]["t_end"]) < max(nodes[i]["t_start"], nodes[j]["t_start"]):
                continue
            ov = int(np.intersect1d(fi, nodes[j]["frames"], assume_unique=True).size)
            if ov <= 0:
                continue
            lj = nodes[j]["length"]
            frac = ov / max(min(li, lj), 1)
            if frac > cfg["overlap_frac_thresh"]:
                if mode == "frames":
                    w = float(ov)
                elif mode == "frac":
                    w = float(frac)
                elif mode == "apt":
                    w = float(min(min(li, lj), ov))
                else:                          # "constant" (original behaviour)
                    w = 1.0
                pairs.append((i, j, w))
    return pairs


def get_cannot_link_pairs(nodes, pinned_edges_false):
    """Unlink (edge pin=False) -> cannot-link super-node pairs.

    The interactive tracker's `u` (Unlink) action asserts "these two tracklets are
    different identities". In this labeling formulation that is a cannot-link:
    x[i,l] + x[j,l] <= 1 for every identity label l -- the exact dual of the must-link
    contraction that `pinned_chains` gets in build_super_nodes.

    Pins are contracted first, so a tracklet id must be mapped through its super-node.
    If both endpoints land in the SAME super-node they are already must-linked, and the
    cannot-link is unsatisfiable -- the ILP would come back infeasible with no useful
    message, so raise here instead, naming the offending pair.
    """
    owner = {}
    for i, nd in enumerate(nodes):
        for t in nd["members"]:
            owner[int(t)] = i

    pairs, skipped = set(), []
    for pair in (pinned_edges_false or []):
        if len(pair) < 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        ia, ib = owner.get(a), owner.get(b)
        if ia is None or ib is None:
            skipped.append((a, b, "dropped or out of range"))
            continue
        if ia == ib:
            raise ValueError(
                f"Contradiction: tracklets {a} and {b} are marked cannot-link "
                f"(pinned_edges_false) but pinned_chains already must-link them into the "
                f"same super-node. Delete the pin or the unlink."
            )
        pairs.add((min(ia, ib), max(ia, ib)))
    for a, b, why in skipped:
        logger.warning("Unlink %d-%d ignored: %s.", a, b, why)
    return sorted(pairs)


# --------------------------------------------------------------------------------------
# The ILP (gurobipy). x[i,l] in {0,1}, l in 0..K (0 = outlier).
# obj = sum_i sum_l unary[i,l] x[i,l]  +  sum_(i,j) a_ij * differ_ij
# --------------------------------------------------------------------------------------
def solve_labeling_ilp(n, K, unary, attract_edges, overlap_pairs, kept_mask, cfg,
                       cannot_link=()):
    m = gp.Model("graph_cut")
    m.Params.OutputFlag = 1
    if cfg.get("time_limit") is not None:
        m.Params.TimeLimit = float(cfg["time_limit"])
    m.Params.MIPGap = float(cfg.get("mip_gap", 0.0))
    m.Params.Threads = int(cfg.get("gurobi_threads", 0))

    L = list(range(K + 1))
    x = m.addVars(n, K + 1, vtype=GRB.BINARY, name="x")
    m.addConstrs((x.sum(i, "*") == 1 for i in range(n)), name="assign")

    # overlap repulsion, weight p = cfg.overlap_penalty (DEFAULT 1.0 = soft): p > 0 pays p per
    # overlapping pair that shares a non-outlier label; p = 0 is the opt-in hard must-not-link.
    # Soft mirrors APT's repulsion and, unlike the hard constraint, lets strong appearance
    # evidence override a spurious temporal overlap -- it scored better on every cohort7 movie.
    overlap_penalty = cfg.get("overlap_penalty")
    overlap_penalty = 1.0 if overlap_penalty is None else float(overlap_penalty)
    logger.info("Overlap repulsion: %s (p=%g)",
                "soft penalty" if overlap_penalty > 0 else "hard must-not-link", overlap_penalty)
    if overlap_penalty <= 0:
        for (i, j, _w) in overlap_pairs:
            for l in range(1, K + 1):
                m.addConstr(x[i, l] + x[j, l] <= 1, name=f"mnl_{i}_{j}_{l}")

    # cannot-link (Unlink / pinned_edges_false): ALWAYS hard. Unlike the overlap
    # repulsion -- a heuristic prior that strong appearance evidence may override --
    # this is a human assertion that the two tracklets are different identities.
    for (i, j) in cannot_link:
        for l in range(1, K + 1):
            m.addConstr(x[i, l] + x[j, l] <= 1, name=f"cnl_{i}_{j}_{l}")
    if cannot_link:
        logger.info("Cannot-link (Unlink) pairs: %d -- hard must-not-link.", len(cannot_link))

    # keep -> non-outlier
    for i in range(n):
        if kept_mask[i]:
            m.addConstr(x[i, 0] == 0, name=f"keep_{i}")

    obj = gp.LinExpr()
    if unary is not None:
        for i in range(n):
            for l in L:
                if unary[i, l] != 0:
                    obj += unary[i, l] * x[i, l]

    # differ_ij = 1 - sum_l same[i,j,l]; same[i,j,l] = x[i,l] & x[j,l] (full linearization)
    for (i, j), a in attract_edges.items():
        same = gp.LinExpr()
        for l in L:
            y = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{i}_{j}_{l}")
            m.addConstr(y <= x[i, l]); m.addConstr(y <= x[j, l])
            m.addConstr(y >= x[i, l] + x[j, l] - 1)
            same += y
        obj += a * (1.0 - same)      # pay attraction weight when labels differ

    if overlap_penalty:              # soft overlap: pay overlap_penalty * weight when a pair shares a label
        for (i, j, w) in overlap_pairs:
            for l in range(1, K + 1):
                y = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"o_{i}_{j}_{l}")
                m.addConstr(y >= x[i, l] + x[j, l] - 1)
                obj += overlap_penalty * w * y

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    logger.info("ILP size: %d variables (%d binary x[i,l], %d continuous y[i,j,l]) | "
                "%d constraints | %d nonzeros.",
                m.NumVars, m.NumBinVars, m.NumVars - m.NumBinVars, m.NumConstrs, m.NumNZs)
    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError(f"ILP found no feasible solution (status {m.Status}). "
                           "Check for a pinned/overlap contradiction.")
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        labels[i] = int(np.argmax([x[i, l].X for l in L]))
    return labels, m.ObjVal


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Cache path: use the APT eval's cached link_pure trk + per-tracklet embedding pkl, so the ILP
# labels the EXACT tracklets + color embeddings the (crashing) APT graph cut uses. Falls back
# to the CSVs when the caches are absent.
# --------------------------------------------------------------------------------------
def _export_trk_csv(trk, out_path, group, t_min, t_max):
    """Write a keypoints CSV (group id t y x parent_id tracklet_id kp..) from a link_pure trk,
    tracklet_id = trk target index, ids incrementing over (target, time) — same format as
    get_pure_tracklets.py, so load_geometry / spatial_overlap / save_solution_keypoints_csv all
    read it unchanged. Returns nlandmarks."""
    if not trk.issparse:
        trk.convert2sparse()
    pTrk, nkp = trk.pTrk, trk.nlandmarks
    all_t, all_tid, all_pose = [], [], []
    for tgt in range(trk.ntargets):
        data = pTrk.data[tgt]                              # [nkp, 2, nframes], col0=x col1=y
        if data is None:
            continue
        valid = np.where(~np.all(np.isnan(data[:, 0, :]), axis=0))[0]
        if len(valid) == 0:
            continue
        all_t.append(int(pTrk.startframes[tgt]) + valid)
        all_tid.append(np.full(len(valid), tgt))
        all_pose.append(data[:, :, valid])                # [nkp, 2, nvalid]
    t_arr = np.concatenate(all_t)
    tid_arr = np.concatenate(all_tid)
    pose = np.concatenate(all_pose, axis=2)               # [nkp, 2, N]
    keep = np.ones(len(t_arr), bool)
    if t_min is not None: keep &= t_arr >= t_min
    if t_max is not None: keep &= t_arr <= t_max
    t_arr, tid_arr, pose = t_arr[keep], tid_arr[keep], pose[:, :, keep]
    N = len(t_arr)
    cols = {"group": group if group is not None else "group", "id": np.arange(1, N + 1),
            "t": t_arr, "y": np.nanmean(pose[:, 1, :], axis=0), "x": np.nanmean(pose[:, 0, :], axis=0),
            "parent_id": np.zeros(N, int), "tracklet_id": tid_arr}
    for kp in range(nkp):
        cols[f"kp{kp}_y"] = pose[kp, 1, :]
        cols[f"kp{kp}_x"] = pose[kp, 0, :]
    order = ["group", "id", "t", "y", "x", "parent_id", "tracklet_id"] + \
            [c for kp in range(nkp) for c in (f"kp{kp}_y", f"kp{kp}_x")]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols)[order].to_csv(out_path, sep=" ", index=False)
    return nkp


def load_from_cache(pure_trk_path, emb_pkl_path, apt_path, kp_csv_out, group, t_min, t_max):
    """From the APT eval caches: tids + per-tracklet color embeddings (pkl), and export the trk
    to a keypoints CSV (kp_csv_out). Returns (tids, E[n,n_ex,D], frames_list, num_keypoints)."""
    import sys, pickle
    if apt_path and apt_path not in sys.path:
        sys.path.insert(0, apt_path)
    import TrkFile
    with open(emb_pkl_path, "rb") as f:
        preds, pred_map, _all_data = pickle.load(f)
    E = np.asarray(preds, np.float32)
    tids = [int(t) for t in np.asarray(pred_map)[:, 1]]
    trk = TrkFile.Trk(str(pure_trk_path))
    if not trk.issparse:
        trk.convert2sparse()
    nkp = _export_trk_csv(trk, kp_csv_out, group, t_min, t_max)
    ss, ee = trk.get_startendframes()
    frames = []
    for tgt in tids:
        fr = np.arange(int(ss[tgt]), int(ee[tgt]) + 1)
        p = trk.gettargetframe(int(tgt), fr)
        vf = fr[~np.all(np.isnan(p[:, 0, :, 0]), axis=0)]
        if t_min is not None: vf = vf[vf >= t_min]
        if t_max is not None: vf = vf[vf <= t_max]
        frames.append(vf)
    logger.info("CACHE inputs: %d embedded tracklets, E=%s, exported keypoints -> %s",
                len(tids), E.shape, kp_csv_out)
    return tids, E, frames, nkp


def resolve_cluster_overlaps(nodes, labels, attract, cfg):
    """APT-faithful post-solve overlap resolution.

    Mirrors deepnet/link_trajectories.py:group_graph_cut (the :4462-4483 drop loop). The ILP
    energy (like APT's alpha-expansion) only *discourages* two temporally-overlapping tracklets
    from sharing an identity -- with soft overlap (p>0, the default/best config) it can still
    place both in one cluster. APT guarantees <=1 tracklet per frame per identity before saving
    apt.trk by dropping the worse-connected overlappers; this reproduces that.

    Per cluster: sort best-connected first (strongest in-cluster motion attraction; longer
    tracklet breaks ties -- the sign-flipped analogue of APT's `np.lexsort((-tlen, link_cost))`),
    then greedily keep nodes, dropping any whose frames overlap (> overlap_resolve_frac of its
    own length) already-kept nodes. Dropped nodes are set to the outlier label 0, so they share
    the outlier fate downstream (singleton if outlier_as_singleton else removed) -- APT routes
    them to `no_group`, which with keep_all_preds=False (its default) means removed; set
    outlier_as_singleton=False to match that exactly.

    Faithful to APT down to the quirks: occupancy is over each node's ACTUAL occupied frames,
    the drop threshold is 5% of length, and a dropped node decrements occupancy only at its
    overlapping frames.
    """
    if not labels.size:
        return labels
    resolve_frac = float(cfg.get("overlap_resolve_frac", 0.05))
    labels = labels.copy()
    n_dropped = 0
    for lab in range(1, int(labels.max()) + 1):
        members = [i for i in range(len(nodes)) if labels[i] == lab]
        if len(members) <= 1:
            continue
        # best in-cluster attraction per node (-inf if isolated -> sorted last, as APT's
        # _best_group_link_cost returns inf cost -> lexsort puts it last)
        link = {}
        for i in members:
            best = -np.inf
            for j in members:
                if j != i:
                    a = attract.get((min(i, j), max(i, j)))
                    if a is not None and a > best:
                        best = a
            link[i] = best
        order = sorted(members, key=lambda i: (-link[i], -nodes[i]["length"]))
        occ = defaultdict(int)
        for i in members:
            for f in nodes[i]["frames"]:
                occ[f] += 1
        for i in order:
            if all(v <= 1 for v in occ.values()):
                break
            ov = [f for f in nodes[i]["frames"] if occ[f] > 1]
            if len(ov) / max(nodes[i]["length"], 1) > resolve_frac:
                for f in ov:
                    occ[f] -= 1
                labels[i] = 0
                n_dropped += 1
    if n_dropped:
        logger.info("Overlap resolution (APT group_graph_cut :4462-4483): dropped %d "
                    "overlapping tracklet-node(s) to outlier.", n_dropped)
    else:
        logger.info("Overlap resolution: no within-cluster overlaps to resolve.")
    return labels


def write_solution_truncated(nodes, labels, attract, csv_path, solution_csv, group, cfg):
    """Overlap resolution by TRUNCATION (frame-level) instead of wholesale tracklet drop.

    Per cluster, greedily fill frames best-tracklet-first (same priority as the drop heuristic:
    strongest in-cluster attraction, longer breaks ties); each tracklet keeps ONLY the frames not
    already occupied by a higher-priority tracklet in its cluster. This guarantees <=1 detection
    per identity per frame while preserving EVERY non-conflicting detection -- only the duplicate
    (physically-impossible) frames are removed, so full frame coverage is retained (unlike the
    wholesale drop, which also discards a losing tracklet's unique frames).

    Writes the solution CSV directly (identity = tracklet_id column, as the scorer expects).
    Outliers (label 0) -> singletons keeping all their frames (or removed if outlier_as_singleton
    is false)."""
    labels = np.asarray(labels)
    by_cluster = defaultdict(list)
    outlier_nodes = []
    for i, lab in enumerate(labels):
        (by_cluster[int(lab)].append(i) if lab >= 1 else outlier_nodes.append(i))

    kept_frames = {}          # tid -> set(frames kept)
    tid_to_cluster = {}       # tid -> identity label
    n_trunc = 0
    for lab, members in by_cluster.items():
        link = {}
        for i in members:
            best = -np.inf
            for j in members:
                if j != i:
                    a = attract.get((min(i, j), max(i, j)))
                    if a is not None and a > best:
                        best = a
            link[i] = best
        order = sorted(members, key=lambda i: (-link[i], -nodes[i]["length"]))
        occupied = set()
        for i in order:
            fr = set(int(f) for f in nodes[i]["frames"])
            keep = fr - occupied
            n_trunc += len(fr) - len(keep)
            occupied |= keep
            for tid in nodes[i]["members"]:
                kept_frames[tid] = kept_frames.get(tid, set()) | keep
                tid_to_cluster[tid] = lab

    next_lab = (int(labels.max()) + 1) if labels.size else 1
    if cfg.get("outlier_as_singleton", True):
        for i in outlier_nodes:
            for tid in nodes[i]["members"]:
                tid_to_cluster[tid] = next_lab
                kept_frames[tid] = set(int(f) for f in nodes[i]["frames"])  # singleton: no self-overlap
                next_lab += 1

    df = pd.read_csv(csv_path, sep=r"\s+")
    tids = df["tracklet_id"].astype(int).to_numpy()
    ts = df["t"].astype(int).to_numpy()
    keep_mask = np.fromiter(
        ((tid in tid_to_cluster) and (t in kept_frames.get(tid, ()))
         for tid, t in zip(tids, ts)), dtype=bool, count=len(df))
    out = df[keep_mask].copy()
    # Preserve the raw detection id + raw tracklet_id before overwriting tracklet_id
    # with the cluster/identity label. The interactive tracker needs original_tracklet_id
    # to reconstruct a chain's raw tracklets (lock-chain / Lock all) and original_id to key
    # embeddings (rank windows) -- the drop path (save_solution_keypoints_csv) already writes both.
    out["original_id"] = out["id"].astype(int)
    out["original_tracklet_id"] = out["tracklet_id"].astype(int)
    out["tracklet_id"] = [tid_to_cluster[int(t)] for t in out["tracklet_id"].astype(int)]
    if group is not None:
        out["group"] = group
    Path(solution_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(solution_csv, sep=" ", index=False)
    logger.info("Truncate resolution: removed %d duplicate-frame detections; wrote %d identities, "
                "%d detections -> %s", n_trunc, out["tracklet_id"].nunique(), len(out), solution_csv)


def main(config_path):
    cfg = load_config(str(config_path))
    print_config(cfg)
    t0 = time.time()

    mode = cfg.get("mode", "apt_faithful")
    assert mode in ("apt_faithful", "correlation_clustering"), f"bad mode {mode!r}"
    csv_path = cfg["csv_path"]
    emb_path = cfg["embeddings_path"]
    emb_prefix = cfg.get("embeddings_attribute_prefix", "emb")
    group = cfg.get("group")
    t_min, t_max = cfg.get("t_min"), cfg.get("t_max")
    num_keypoints = cfg.get("num_keypoints", 4)
    # Prototype-count mode is explicit (`prototype_mode`: "thresh" | "known"), with a single
    # `num_animals` used by both. "known" -> K = num_animals (maxclust, fraction filter off).
    # "thresh" -> cluster at cluster_threshold, then lower min_cluster_frac until at least
    # num_animals prototypes survive (see get_prototypes). Legacy fallback: if prototype_mode
    # is absent, the old num_tracks key still selects the mode (null -> thresh, set -> known).
    num_animals = cfg.get("num_animals")
    proto_mode = cfg.get("prototype_mode")
    if proto_mode is not None:
        assert proto_mode in ("thresh", "known"), f"bad prototype_mode {proto_mode!r} (thresh|known)"
        assert num_animals is not None, "prototype_mode set -> num_animals is required in the config"
        num_tracks = int(num_animals) if proto_mode == "known" else None
    else:
        num_tracks = cfg.get("num_tracks")
        proto_mode = "known" if num_tracks else "thresh"
    logger.info("Prototype mode: %s (num_animals=%s).", proto_mode, num_animals)
    n_ex = cfg.get("cluster_n_samples", 25)
    seed = cfg.get("seed", 0)
    solution_csv = Path(cfg["solution_csv"]).expanduser()
    solution_csv.parent.mkdir(parents=True, exist_ok=True)

    dropped = {int(t) for t in (cfg.get("pinned_nodes_false") or [])}
    kept = {int(t) for t in (cfg.get("pinned_nodes_true") or [])}
    dropped -= kept                                   # keep overrides drop
    pinned_chains = cfg.get("pinned_chains") or []

    # 1) inputs: prefer the APT eval's caches (link_pure pure_tracklets.trk + per-tracklet color
    #    embedding pkl) so the ILP labels the EXACT tracklets + embeddings the APT graph cut uses;
    #    fall back to the CSVs when a cache is absent. csv_path is (re)written from the trk in the
    #    cache case, so load_geometry / spatial_overlap / save_solution_keypoints_csv are unchanged.
    rng = np.random.default_rng(seed)
    pure_trk = cfg.get("pure_trk_path")
    emb_pkl = cfg.get("embeddings_pkl_path")
    if pure_trk and emb_pkl and Path(pure_trk).exists() and Path(emb_pkl).exists():
        tids, E, frames, num_keypoints = load_from_cache(
            pure_trk, emb_pkl, cfg.get("apt_path"), csv_path, group, t_min, t_max)
    else:
        logger.info("CSV inputs: %s + %s", csv_path, emb_path)
        tracklets = load_tracklets(csv_path, emb_path, emb_prefix, t_min, t_max)
        tids, E, frames, _ = sample_per_tracklet(tracklets, n_ex, rng)
    geom = load_geometry(csv_path, num_keypoints, t_min, t_max)
    nodes = build_super_nodes(tids, E, frames, geom, dropped, kept, pinned_chains, n_ex, rng)
    n = len(nodes)
    logger.info("%d tracklets -> %d after drop -> %d super-nodes (%d pinned chains).",
                len(tids), len(tids) - len(dropped), n, len(pinned_chains))

    cannot_link = get_cannot_link_pairs(nodes, cfg.get("pinned_edges_false"))
    if cannot_link:
        logger.info("%d unlink pair(s) from pinned_edges_false -> cannot-link.", len(cannot_link))

    overlap_pairs = get_overlap_pairs(nodes, cfg) if cfg.get("enforce_overlap", True) else []
    logger.info("Temporally-overlapping pairs (>%.0f%% of shorter node): %d -- repulsed either "
                "softly or hard, see the next overlap line.",
                100 * cfg["overlap_frac_thresh"], len(overlap_pairs))

    # 2) build the ILP terms for the chosen mode
    if mode == "apt_faithful":
        apt_npz = cfg.get("apt_unary_npz")
        if apt_npz and Path(apt_npz).exists():
            # APT's OWN cluster centers + unary (get_id_cluster_centers / get_graph_cut_unary),
            # precomputed to npz. unary_vals is per trk target; map to super-nodes (mean over
            # members) and scale by unary_wt exactly as APT (k_way_graph_cut(unary_vals*unary_wt)).
            z = np.load(apt_npz)
            uv, K = z["unary_vals"], int(z["K"])
            unary = np.array([uv[nd["members"]].mean(axis=0) for nd in nodes]) * cfg["unary_wt"]
            logger.info("apt_faithful: APT unary from %s (K=%d).", apt_npz, K)
        else:
            embs_stack = [nd["embs"] for nd in nodes]
            dist_diag = intra_spread(embs_stack)
            oa_tid = spatial_overlap(csv_path, num_keypoints, t_min, t_max)   # APT get_overlap_value
            oa = np.array([np.mean([oa_tid.get(t, 0.0) for t in nd["members"]]) for nd in nodes])
            logger.info("Spatial overlap oa: median=%.3f, frac<%.2f=%.2f",
                        float(np.median(oa)), cfg["centre_max_overlap"],
                        float(np.mean(oa < cfg["centre_max_overlap"])))
            centres = get_prototypes(nodes, dist_diag, oa, num_tracks, cfg)
            K = len(centres)
            logger.info("apt_faithful: %d prototype identities.", K)
            unary = get_unary(nodes, centres, cfg) * cfg["unary_wt"] if cfg.get("use_id_embedding_distance", True) \
                else np.zeros((n, K + 1))
        attract = get_motion_pairwise(nodes, cfg) if cfg.get("use_spatial_distance", True) else {}
    else:
        D = id_distance_matrix(np.stack([nd["embs"] for nd in nodes]))
        if num_tracks:
            K = int(num_tracks)
        else:                                          # threshold-derived upper bound on #labels
            F = fcluster(linkage(squareform(D, checks=False), "average"),
                         cfg["cluster_threshold"], criterion="distance")
            K = int(F.max())
        logger.info("correlation_clustering: K=%d labels.", K)
        unary = None
        attract = get_correlation_edges(D, cfg) if cfg.get("use_id_embedding_distance", True) else {}
    logger.info("Attraction edges: %d", len(attract))

    # 3) solve
    kept_mask = np.array([nd["kept"] for nd in nodes])
    labels, obj = solve_labeling_ilp(n, K, unary, attract, overlap_pairs, kept_mask, cfg,
                                     cannot_link=cannot_link)
    logger.info("ILP objective %.3f; %d non-outlier identities used.",
                obj, len({l for l in labels if l >= 1}))

    # 3b) post-solve overlap resolution so co-clustered tracklets don't temporally overlap in the
    #     final solution. overlap_resolution:
    #       drop     (default) -> wholesale-drop the worse tracklet to outlier (APT group_graph_cut
    #                             :4462-4483). Lossy: discards the dropped tracklet's unique frames.
    #       truncate           -> keep the tracklet, remove only its conflicting frames (frame-level).
    #                             Preserves every non-conflicting detection; writes the CSV directly.
    resolution = cfg.get("overlap_resolution", "drop")
    if cfg.get("resolve_overlaps", True) and resolution == "truncate":
        write_solution_truncated(nodes, labels, attract, csv_path, solution_csv, group, cfg)
        logger.info("Done in %.1fs (truncate resolution).", time.time() - t0)
        return

    if cfg.get("resolve_overlaps", True):
        labels = resolve_cluster_overlaps(nodes, labels, attract, cfg)
        logger.info("After overlap resolution: %d non-outlier identities used.",
                    len({l for l in labels if l >= 1}))

    # 4) labels -> per-identity chains -> CSV. Outlier (0) tracklets become singletons if asked.
    tid_to_cluster, next_lab = {}, K + 1
    for nd, lab in zip(nodes, labels):
        if lab == 0 and cfg.get("outlier_as_singleton", True):
            for tid in nd["members"]:
                tid_to_cluster[tid] = next_lab; next_lab += 1
        elif lab == 0:
            continue                                   # drop outliers from the solution
        else:
            for tid in nd["members"]:
                tid_to_cluster[tid] = int(lab)

    t_start_of = {t: geom[t]["t_start"] if t in geom else 0 for t in tid_to_cluster}
    by_cluster = defaultdict(list)
    for tid, lab in tid_to_cluster.items():
        by_cluster[lab].append(tid)
    G = nx.DiGraph()
    for tids_c in by_cluster.values():
        s = sorted(tids_c, key=lambda t: t_start_of[t])
        G.add_nodes_from(s)
        for a, b in zip(s[:-1], s[1:]):
            G.add_edge(a, b)

    save_solution_keypoints_csv(G, csv_path, solution_csv, group=group)
    logger.info("Done in %.1fs: %d tracklets -> %d identities -> %s",
                time.time() - t0, len(tid_to_cluster), len(by_cluster), solution_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="path to configs_graph_cut.yaml")
    args = parser.parse_args()
    main(args.config)
