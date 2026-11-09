"""infer_graph_cut.py — APT-style graph-cut tracklet stitching, formulated as an exact ILP (gurobipy).

This one reproduces APT's *labeling* graph cut
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

Method: prototypes from hierarchical clustering of the clean, long (>50f) subset ->
per-tracklet UNARY (dist to nearest prototype) + Potts MOTION pairwise (attraction between
temporally-local, motion-consistent tracklets) + temporal-overlap repulsion + K identities.
(A second `correlation_clustering` mode once existed, selected by a `mode` config key; both
were removed -- the mode was never used and its pairwise id-embedding distance matrix cannot
admit the embedding-less tracklets this script now includes.)

Embedding eligibility: upstream computes id embeddings only for tracklets of >= 6 frames.
Those long tracklets carry the appearance UNARY; every tracklet, however short, still
contributes the MOTION pairwise and appears in the output. So the node set is the full
keypoints CSV (load_all_tracklets), not the embedding file -- an inner join there would
silently delete the short fragments (84% of tracklet ids on cohort7). A pinned_chains entry
naming an embedding-less tracklet is therefore honoured; a pin also forces its endpoints
non-outlier (pin_implies_keep, default True).

Faithfulness notes:
  * EVERY input tracklet is labeled. In APT the graph cut labels all pure tracklets; the >=6
    frame cutoff is only APT's embedding-eligibility, and the >50 cutoff only selects prototype
    centres.
  * Occlusion: APT weights several terms by pTrkTag occlusion tags. The tracking_library CSV has
    no occlusion column, so occ=0 -> APT's occlusion terms vanish cleanly (unary unchanged,
    centre weighting uniform, occ<0.2 filter passes all).
  * Motion pairwise here is a faithful-in-spirit reimplementation of get_motion_links +
    get_graph_cut_pairwise from the CSV keypoints (endpoint centroid + single-step velocity,
    gap-penalised cost, top-k symmetric, a = max(0, 4 - m/len_div)). d_norm (APT's maxcost*2)
    is approximated by 2x the median per-frame centroid step.
  * delete_short: AFTER the ILP labels become chains, any final chain occupying
    <= delete_short_maxframes frames (default 10) is deleted from the output -- APT's
    link_trklet_id runs delete_short on the grouped ids unless keep_all_preds is set (its
    default is to delete; maxframes_delete=10). This is length-based and outlier-blind, so a
    long unstitched singleton survives while short fragments are removed. Set
    delete_short_maxframes: 0 to disable (~ APT keep_all_preds=True).

Same solver contract as infer_clusters.py: single positional config arg; honors
pinned_nodes_false (drop -> excluded), pinned_nodes_true (keep -> non-outlier),
pinned_chains (pin edge -> must-link / same identity); writes via save_solution_keypoints_csv.

Freezing one identity takes TWO keys, because "what is in identity l" is a property of the
whole assignment, not of that identity's own nodes:
  * pinned_nodes_identity: {tracklet: animal}   -> x[i,l] == 1, who IS in it
  * identity_closed: {animal: [[t0, t1], ...]}  -> x[j,l] == 0 for every OTHER node inside the
                                                   window, i.e. nobody else is
Together they say: over the frames [t0, t1] I inspected, identity l is exactly these tracklets.
The window is mandatory -- a global closure also asserts the animal is absent everywhere else.
Must-link (pinned_chains) alone does neither: it only asserts that a chain's members share a
label, leaving both which label and who else may take it to the solve. See get_closed_identities.

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
from utils import (
    id_distance_matrix,
    load_all_tracklets,
    load_geometry,
    build_super_nodes,
    intra_spread,
    spatial_overlap,
    nodes_mostly_inside,
    load_from_cache,
    resolve_cluster_overlaps,
    delete_short_chains,
    clip_window,
    outliers_path,
    write_cluster0,
    label_mapper,
    raw_tracklet_nodes,
    write_solution_truncated,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# apt_faithful: prototypes (get_id_cluster_centers) + unary (get_graph_cut_unary)
# --------------------------------------------------------------------------------------
def get_prototypes(nodes, dist_diag, oa, num_tracks, cfg):
    """Hierarchical clustering of the clean, long subset -> occupancy/length-weighted mean
    prototype centres. Mirrors get_id_cluster_centers (occ term inactive: occ=0)."""
    E_mean = np.stack([nd["embs"].mean(axis=0) for nd in nodes])   # [n, D]
    lengths = np.array([nd["length"] for nd in nodes], float)
    # `close_thresh_value` (optional) fixes the spread threshold instead of re-deriving it as a
    # percentile of THIS node set. Needed when clustering a handful of pooled super-nodes: over ~12
    # nodes the 95th percentile always drops the single most variable one, i.e. a whole animal.
    close_thresh = cfg.get("close_thresh_value")
    if close_thresh is None:
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

    # min_cluster_frac is a fraction of the MOVIE LENGTH in frames, as in APT
    # get_id_cluster_centers (`g_sz / sum(len_mov) < 0.05`, len_mov = max(ee) - min(ss)).
    # It used to be a fraction of the summed length of all embedded tracklets, which with ~12
    # animals is ~10x larger and dropped real animals' clusters (cohort7 _1029: 6 prototypes
    # instead of APT's 12 with no num_animals).
    total_len = float(max(nd["t_end"] for nd in nodes) - min(nd["t_start"] for nd in nodes))

    # Effective min-cluster fraction. In threshold mode, if `num_animals` is set and the
    # configured min_cluster_frac admits FEWER than num_animals prototypes, lower the fraction
    # just enough to admit at least num_animals clusters -- i.e. down to the size (frac of movie
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


def load_fixed_prototypes(proto_path, nodes, cfg):
    """Identity prototypes carried over from an already-fixed movie of the same session.

    Written by extract_prototypes.py as the per-chain mean embedding of a FIXED solution --
    the same quantity get_prototypes computes in-movie (its length-weighted mean of per-tracklet
    means is exactly the flat mean over detections), so a loaded centre is interchangeable with a
    derived one and get_unary needs no change.

    Skipping get_prototypes also skips intra_spread and spatial_overlap, which exist only to pick
    the "clean" prototype subset -- so this path is cheaper as well as consistent.

    Because label l always means the same animal, an animal ABSENT from this movie simply leaves
    label l unused; nothing needs to be told how many animals are present.
    """
    z = np.load(proto_path, allow_pickle=True)
    centres = z["centres"].astype(np.float64)
    ids = np.asarray(z["animal_ids"] if "animal_ids" in z.files
                     else np.arange(1, len(centres) + 1), dtype=int)

    # `prototypes_exclude`: animals KNOWN to be absent from this movie. Optional, and normally
    # unnecessary -- an absent animal's label simply costs more than the right one for every
    # node, so the ILP leaves it unused on its own. Its value is removing the small phantom
    # chains a weak prototype can still attract. The cost of using it is that a wrong guess is
    # SILENT: the animal's tracklets are forced to their next-best label instead of showing up
    # as a suspiciously tiny cluster, which is the signal absence would otherwise announce
    # itself with. Prefer solving once, reading the per-animal frame mass, then declaring.
    excl = {int(a) for a in (cfg.get("prototypes_exclude") or [])}
    if excl:
        unknown = excl - set(ids.tolist())
        if unknown:
            raise ValueError(f"prototypes_exclude names animal(s) {sorted(unknown)} that are not "
                             f"in {proto_path} (has {ids.tolist()})")
        keep = np.array([int(a) not in excl for a in ids])
        if not keep.any():
            raise ValueError("prototypes_exclude removed every prototype")
        centres, ids = centres[keep], ids[keep]
        logger.info("prototypes_exclude: dropped animal(s) %s -> %d prototype(s) remain.",
                    sorted(excl), len(ids))
    K = len(centres)

    emb_idx = [i for i, nd in enumerate(nodes) if nd["embs"] is not None]
    if not emb_idx:
        raise ValueError("no tracklet has embeddings -- the appearance unary would be empty")
    D = nodes[emb_idx[0]]["embs"].shape[1]
    if centres.shape[1] != D:
        raise ValueError(f"{proto_path}: prototype dim {centres.shape[1]} != embedding dim {D} "
                         f"-- these came from different encoders")
    logger.info("apt_faithful: %d FIXED prototype identities from %s (animals %s).",
                K, proto_path, list(map(int, ids)))

    # Drift alarm. outlier_unary is an ABSOLUTE threshold on mean sample-to-centre distance, so
    # prototypes from another movie are only usable if this movie's distances land in the same
    # range. If a day's appearance shifted (lighting, camera, the animals themselves), every
    # distance inflates and the ILP quietly sends the whole movie to the outlier label -- an
    # empty solution with no error. Report the distribution rather than let that happen silently.
    d_min = np.array([np.linalg.norm(nodes[i]["embs"][:, None, :] - centres[None, :, :],
                                     axis=-1).mean(axis=0).min() for i in emb_idx])
    thr = float(cfg["outlier_unary"])
    frac = float(np.mean(d_min > thr))
    logger.info("Prototype fit over %d embedded node(s): dist-to-nearest median=%.2f p90=%.2f "
                "p99=%.2f max=%.2f; %.1f%% above outlier_unary=%.2f.",
                len(emb_idx), float(np.median(d_min)), float(np.percentile(d_min, 90)),
                float(np.percentile(d_min, 99)), float(d_min.max()), 100 * frac, thr)
    if frac > 0.5:
        logger.warning("%.0f%% of embedded nodes are further than outlier_unary from EVERY "
                       "prototype -- the prototypes likely do not fit this movie (different "
                       "encoder, or appearance drift). Re-derive them, or raise outlier_unary.",
                       100 * frac)
    return centres, K, ids


def get_unary(nodes, centres, cfg):
    """u[i, l] for l in 0..K. u[i,0] = outlier_unary; u[i,k] = mean sample-to-centre distance.

    A node with no embeddings (short tracklet) has NO appearance evidence, so it contributes
    no unary at all -- its label is decided by the motion pairwise. Note a row of exact zeros
    is not usable as-is: with u[i,0] = outlier_unary * unary_wt (20 by default) still in
    place, zeros on the identity labels would make every identity strictly cheaper than
    outlier, i.e. silently FORCE such a node into some track. Zeroing the whole row instead
    leaves all K+1 labels tied, and Gurobi would then break the tie arbitrarily -- a fragment
    with no motion link would land in a random identity. So the row is zero except for a tiny
    `no_emb_outlier_bias` on the identity labels: absent any attraction the node falls to
    outlier, while any real motion attraction (a up to pairwise_attract_max = 4) dwarfs the
    bias and pulls it into the track.
    """
    n, K = len(nodes), len(centres)
    U = np.zeros((n, K + 1))
    U[:, 0] = cfg["outlier_unary"]
    bias = float(cfg.get("no_emb_outlier_bias", 1e-3))
    n_noemb = 0
    for i, nd in enumerate(nodes):
        e = nd["embs"]                                  # [n_ex, D] or None
        if e is None:
            U[i, :] = 0.0
            U[i, 1:] = bias
            n_noemb += 1
            continue
        for k, c in enumerate(centres):
            U[i, k + 1] = np.linalg.norm(e - c[None, :], axis=1).mean()
    if n_noemb:
        logger.info("Unary: %d node(s) without embeddings -> no appearance term "
                    "(outlier bias %g; motion pairwise decides).", n_noemb, bias)
    # `outlier_len_scale: {len0: 100, pow: 0.5}` (the DEFAULT since 2026-10-27; `false` turns it off): the unary is
    # per tracklet, not per frame, so sending a 10k-frame tracklet to outlier would cost the same as a 10-frame one,
    # and the solver could discard a long, correctly embedded tracklet that overlaps wrongly labelled fragments
    # (cohort7 step 8, 2026-10-05). Scaling the outlier cost by max(1, (length / len0) ** pow) makes discarding long
    # tracklets expensive. On 22 cohort7 movies (mode 4) it raised open-arena IDF1 0.968 -> 0.971.
    ls = cfg.get("outlier_len_scale", {"len0": 100, "pow": 0.5})
    if ls:
        lengths = np.array([nd["length"] for nd in nodes], float)
        f = np.maximum(1.0, (lengths / float(ls.get("len0", 100))) ** float(ls.get("pow", 0.5)))
        U[:, 0] *= f
        logger.info("Outlier cost scaled by max(1, (length / %g) ** %g): %d node(s) scaled, max factor %.1f.",
                    float(ls.get("len0", 100)), float(ls.get("pow", 0.5)), int((f > 1).sum()), float(f.max()))
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

    # Candidate successors by a gap-bounded sweep instead of the full i x j double loop: only
    # j with t_start in [t_end(i)+1, t_end(i)+max_gap] can pass, so sort nodes by t_start and
    # binary-search that window. Identical output to the O(n^2) form (same two conditions,
    # i before j and 0 < g <= max_gap), but it matters now that EVERY tracklet is a node --
    # embedding-less fragments took the count from ~2.4k to ~14.7k on cohort7, i.e. 217M
    # iterations for the double loop vs a few million here.
    order = np.argsort([nd["t_start"] for nd in nodes], kind="stable")
    ts_sorted = np.array([nodes[i]["t_start"] for i in order])

    cand = defaultdict(list)                            # i -> list of (a_ij, j)
    for i in range(n):
        ni = nodes[i]
        lo = int(np.searchsorted(ts_sorted, ni["t_end"] + 1, "left"))
        hi = int(np.searchsorted(ts_sorted, ni["t_end"] + max_gap, "right"))
        for k in range(lo, hi):
            j = int(order[k])
            if i == j:
                continue
            nj = nodes[j]
            g = nj["t_start"] - ni["t_end"]
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

    Shared-frame counts come from a frame -> occupants index rather than an n^2 sweep of
    np.intersect1d: a pair can only overlap if it co-occurs in some frame, so bucket every
    node's frames and count co-occurrences per bucket. Cost is sum_t C(k_t, 2) with k_t the
    occupancy of frame t (~8 on cohort7), against 108M pair tests for the double loop once
    embedding-less tracklets join the node set. The span pre-filter is no longer needed --
    bucketing only ever visits frames a node actually occupies, which is also exactly the
    behaviour the contracted-super-node note above calls for.
    """
    n = len(nodes)
    # overlap_weight_mode scales the per-pair repulsion by HOW MUCH two tracklets overlap
    # (APT's repulsion is length/overlap-scaled `-min(min_len, ovv*min_len)`, not flat):
    #   constant -> 1.0            (original: flat penalty per overlapping pair)
    #   frames   -> shared frames  (a 20-frame overlap repels 20x a 1-frame overlap)
    #   frac     -> shared frames / shorter length  (in [0,1])
    #   apt      -> min(shorter length, shared frames)  (APT's own form)
    mode = cfg.get("overlap_weight_mode", "constant")

    occupants = defaultdict(list)                 # frame -> node indices (ascending)
    for i in range(n):
        for f in nodes[i]["frames"]:
            occupants[int(f)].append(i)
    shared = defaultdict(int)                     # (i<j) -> shared frame count
    for lst in occupants.values():
        if len(lst) < 2:
            continue
        for a in range(len(lst)):
            ia = lst[a]
            for b in range(a + 1, len(lst)):
                shared[(ia, lst[b])] += 1         # lst ascending -> key already (min, max)

    pairs = []
    for (i, j), ov in shared.items():
        li, lj = nodes[i]["length"], nodes[j]["length"]
        frac = ov / max(min(li, lj), 1)
        if frac > cfg["overlap_frac_thresh"]:
            if mode == "frames":
                w = float(ov)
            elif mode == "frac":
                w = float(frac)
            elif mode == "apt":
                w = float(min(min(li, lj), ov))
            else:                                 # "constant" (original behaviour)
                w = 1.0
            pairs.append((i, j, w))
    pairs.sort(key=lambda p: (p[0], p[1]))        # deterministic order (dict iteration)
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


def get_identity_window_forbidden(nodes, animal_ids, cfg):
    """`identity_start_frames` / `identity_end_frames` -> [(node, label)] pairs fixing x[i,l] = 0.

    Animals are added to and removed from the arena mid-recording. An identity that has not yet
    been introduced, or has already been taken out, cannot be carried by a tracklet outside its
    active window:

        identity_end_frames:   {12: 10000}          # animal 12 removed at t=10000
        identity_start_frames: { 5: 30000}          # animal  5 introduced at t=30000
        identity_forbidden_spans: {12: [[3000, 5000]]}   # ...and not present in between

    A tracklet is forbidden when at least `identity_span_overlap_frac` (default 0.5) of its own
    extent falls inside the forbidden region. All three keys use the same test -- an end is the
    region (e, +inf), a start is (-inf, s) -- so a tracklet mostly outside the active window is
    banned while one only clipping the boundary keeps its label.

    These are PROHIBITIONS, not deletions. The assignment constraint still holds, so a forbidden
    node takes some other label -- another identity, or 0 (outlier) and thence cluster 0, where
    the displacement is visible and you can check the frame was right. Contrast a drop, which
    removes the node from the problem entirely.

    Requires fixed prototypes: without them the labels are arbitrary per-movie indices and naming
    an animal means nothing, so the keys are refused rather than silently constraining whichever
    identity happens to occupy that index. NB the animal id is NOT the label index --
    `prototypes_exclude: [3,5,7]` leaves animal 12 at index 9 -- so keys are mapped through the
    same animal_ids ordering `label_mapper` uses.
    """
    ends = cfg.get("identity_end_frames") or {}
    starts = cfg.get("identity_start_frames") or {}
    spans = cfg.get("identity_forbidden_spans") or {}
    # identity_absent: the animal is OUT of the arena over [t0, t1] (t1 null = not back yet).
    # Written by the interactive tracker's Exit/Entry marks. For the ILP it is the same
    # prohibition as a forbidden span; the difference (absent vs present-but-unassignable) only
    # matters to the tracker's gap filling.
    absent = cfg.get("identity_absent") or {}
    if not ends and not starts and not spans and not absent:
        return []
    if animal_ids is None:
        raise ValueError(
            "identity_start_frames / identity_end_frames require `prototypes_path`: without "
            "fixed prototypes the ILP labels are arbitrary per-movie indices, so naming an "
            "animal is meaningless")
    label_of = {int(a): i + 1 for i, a in enumerate(animal_ids)}

    # How much of a tracklet must fall inside the forbidden region for its label to be banned.
    # One dial spanning both candidate rules: 1.0 = only WHOLLY-contained tracklets (the
    # original rule (a)); ->0 = any overlap at all (rule (b)).
    #
    # Neither extreme works on real data. At 1.0 a 10-frame span missed a 15-frame tracklet
    # that overhung it by 4 frames -- the thing the user was pointing at. At "any overlap" a
    # ONE-frame span would have banned an 11,630-frame tracklet merely passing through, i.e.
    # displaced an entire identity to excise one frame. 0.5 catches the 15-frame tracklet
    # (14/15 inside) and spares both the 514-frame neighbour (1/514) and the 11,630-frame one.
    frac_req = float(cfg.get("identity_span_overlap_frac", 0.5))
    INF = 10 ** 12          # stands in for an open-ended region (start/end are semi-infinite)

    def _forbidden_in(t0, t1):
        return nodes_mostly_inside(nodes, t0, t1, frac_req)

    def _label(a, key):
        a = int(a)
        if a not in label_of:
            raise ValueError(f"{key} names animal {a}, which is not in this prototype set "
                             f"{sorted(label_of)} (excluded, or not in the file)")
        return label_of[a]

    forbidden = []
    for a, t_end in ends.items():
        l, t_end = _label(a, "identity_end_frames"), int(t_end)
        after, partial = _forbidden_in(t_end + 1, INF)
        forbidden.extend((i, l) for i in after)
        logger.info("identity_end_frames: animal %s (label %d) ends at t=%d -> label forbidden "
                    "on %d node(s) at least %.0f%% after it; %d overlap less and keep it.",
                    a, l, t_end, len(after), 100 * frac_req, partial)
        if not after:
            logger.warning("identity_end_frames: animal %s ends at t=%d but NO node starts after "
                           "it -- the constraint does nothing. If this chain was locked, its "
                           "tracklets are contracted into one super-node whose t_start precedes "
                           "the cut; mark the end BEFORE Lock all.", a, t_end)
    for a, t0 in starts.items():
        l, t0 = _label(a, "identity_start_frames"), int(t0)
        before, partial = _forbidden_in(-INF, t0 - 1)
        forbidden.extend((i, l) for i in before)
        logger.info("identity_start_frames: animal %s (label %d) starts at t=%d -> label "
                    "forbidden on %d node(s) at least %.0f%% before it; %d overlap less and "
                    "keep it.", a, l, t0, len(before), 100 * frac_req, partial)
        if not before:
            logger.warning("identity_start_frames: animal %s starts at t=%d but NO node ends "
                           "before it -- the constraint does nothing (mark it BEFORE Lock all).",
                           a, t0)

    for a, ranges in spans.items():
        l = _label(a, "identity_forbidden_spans")
        for rng in ranges:
            t0, t1 = int(rng[0]), int(rng[1])
            if t1 < t0:
                t0, t1 = t1, t0
            # Rule (a) generalised: a tracklet is forbidden only if it lies WHOLLY inside the
            # span. A start frame s is the span (-inf, s) and an end frame e is (e, +inf), and
            # "wholly outside the active window" is exactly "wholly inside the forbidden span",
            # so all three keys are the same predicate. A tracklet merely PASSING THROUGH the
            # span keeps its label -- to excise part of one, cut it (`c`) first, which is what
            # creates the tracklet boundary the ILP can act on.
            inside, partial = _forbidden_in(t0, t1)
            forbidden.extend((i, l) for i in inside)
            logger.info("identity_forbidden_spans: animal %s (label %d) forbidden on [%d, %d] "
                        "-> %d node(s) at least %.0f%% inside; %d overlap less and keep it.",
                        a, l, t0, t1, len(inside), 100 * frac_req, partial)
            if not inside:
                logger.warning("identity_forbidden_spans: animal %s span [%d, %d] contains no "
                               "whole node -- the constraint does nothing. A tracklet spanning "
                               "it is not forbidden; cut it first, and mark BEFORE Lock all.",
                               a, t0, t1)

    for a, ranges in absent.items():
        l = _label(a, "identity_absent")
        for rng in ranges:
            t0 = int(rng[0])
            t1 = INF if rng[1] is None else int(rng[1])
            inside, partial = _forbidden_in(t0, t1)
            forbidden.extend((i, l) for i in inside)
            logger.info("identity_absent: animal %s (label %d) absent on [%d, %s] -> %d node(s) "
                        "at least %.0f%% inside; %d overlap less and keep it.",
                        a, l, t0, "end" if rng[1] is None else t1, len(inside),
                        100 * frac_req, partial)

    # A window that excludes everything is a typo, not a constraint -- catch it here rather than
    # letting Gurobi return an unhelpful INFEASIBLE.
    by_label = defaultdict(set)
    for i, l in forbidden:
        by_label[l].add(i)
    for l, idx in by_label.items():
        if len(idx) == len(nodes):
            a = [k for k, v in label_of.items() if v == l][0]
            raise ValueError(f"identity window for animal {a} forbids its label on EVERY node "
                             f"-- check identity_start_frames / identity_end_frames / "
                             f"identity_absent for it")
    return sorted(set(forbidden))


def get_forced_identities(nodes, animal_ids, cfg):
    """`pinned_nodes_identity: {tracklet: animal}` -> [(node, label)] pairs fixing x[i, l] == 1.

    The positive counterpart of everything else. Until now you could say a tracklet IS some
    identity (pinned_nodes_true), is the outlier (pinned_nodes_outlier), or is NOT a given
    identity in a frame range (identity_* windows) -- but never WHICH identity it is.
    `pinned_chains` does not fill the gap: it is must-link only, asserting that its members share
    an identity while leaving the appearance unary to choose which. When that choice is wrong the
    whole chain lands in another animal, and if that animal already holds those frames the
    truncation erases every member, so the pin produces nothing visible at all.

    Requires fixed prototypes -- without them a label is an arbitrary per-movie index -- and maps
    animal -> label through animal_ids, so `prototypes_exclude` offsets are handled (with
    [3,5,7] excluded, animal 12 is label 9).

    A whole must-link group inherits the constraint from any member: they share a super-node, so
    forcing one forces all. Naming two different animals inside one group is a contradiction and
    raises here rather than returning an unexplained INFEASIBLE.
    """
    want = cfg.get("pinned_nodes_identity") or {}
    if not want:
        return []
    if animal_ids is None:
        raise ValueError("pinned_nodes_identity requires `prototypes_path`: without fixed "
                         "prototypes the ILP labels are arbitrary per-movie indices")
    label_of = {int(a): i + 1 for i, a in enumerate(animal_ids)}

    owner = {}
    for i, nd in enumerate(nodes):
        for t in nd["members"]:
            owner[int(t)] = i

    by_node, missing = {}, []
    for t, a in want.items():
        t, a = int(t), int(a)
        if a not in label_of:
            raise ValueError(f"pinned_nodes_identity assigns tracklet {t} to animal {a}, which is "
                             f"not in this prototype set {sorted(label_of)}")
        i = owner.get(t)
        if i is None:
            missing.append(t)          # dropped, or outside the frame window
            continue
        prev = by_node.get(i)
        if prev is not None and prev[0] != a:
            raise ValueError(f"pinned_nodes_identity is contradictory: tracklets {prev[1]} and {t} "
                             f"are must-linked into one super-node but are assigned to animals "
                             f"{prev[0]} and {a}")
        by_node[i] = (a, t)
    if missing:
        logger.warning("pinned_nodes_identity: %d tracklet(s) absent from the node set "
                       "(dropped or out of range), ignored: %s", len(missing), sorted(missing)[:20])
    out = [(i, label_of[a]) for i, (a, _t) in by_node.items()]
    for i, (a, t) in sorted(by_node.items()):
        logger.info("pinned_nodes_identity: tracklet %d -> animal %d (label %d), forcing the "
                    "whole super-node (%d member(s)).", t, a, label_of[a], len(nodes[i]["members"]))
    return out


def get_barrier_sides(nodes, geom, cfg):
    """`barrier: {axis: x, position: 2406, state: closed}` -> per-node side (0=left, 1=right,
    None=unconstrained), or None when the barrier is open.

    A physical divider means an animal cannot be on both sides, so no IDENTITY may span it. That
    is a partition of the label set, and the useful part is that it needs no classifier: side is
    read off the coordinates. On cohort7 _1959 the divider shows up as a 506 px band at
    x in [2153, 2659] holding 227 of 750,939 detections (0.03%), and ZERO of the 20,822 raw
    tracklets cross it -- so the partition is exact and covers every node, including the 74.8%
    of tracklets that have no embeddings and could not be classified any other way.

    Encoded in solve_labeling_ilp with one binary s[l] per label ("label l lives on the right"),
    which lets the ILP CHOOSE the side assignment rather than making the user declare which
    animal is where:

        left node i :  x[i,l] + s[l] <= 1
        right node j:  x[j,l]        <= s[l]

    n*K constraints and K binaries -- negligible beside the model's ~11M rows.

    A super-node straddling the barrier is a must-link (pinned_chains) across it, i.e. a human
    pin asserting one identity on both sides. Under `closed` that is precisely the error the
    barrier exists to catch, so it raises with the members named rather than being skipped.
    """
    b = cfg.get("barrier") or {}
    if not b or str(b.get("state", "open")).lower() != "closed":
        return None
    axis = str(b.get("axis", "x")).lower()
    if axis not in ("x", "y"):
        raise ValueError(f"barrier.axis must be 'x' or 'y', got {axis!r}")
    if b.get("position") is None:
        raise ValueError("barrier.state is 'closed' but barrier.position is not set")
    pos = float(b["position"])
    kmin, kmax = f"{axis}_min", f"{axis}_max"

    sides, straddle, no_geom = [], [], 0
    for i, nd in enumerate(nodes):
        gg = [geom[t] for t in nd["members"] if t in geom]
        if not gg:
            sides.append(None); no_geom += 1; continue
        lo = min(g[kmin] for g in gg); hi = max(g[kmax] for g in gg)
        if hi < pos:
            sides.append(0)
        elif lo > pos:
            sides.append(1)
        else:
            sides.append(None); straddle.append((i, sorted(nd["members"])[:8], lo, hi))
    if straddle:
        raise ValueError(
            f"barrier is closed at {axis}={pos:g} but {len(straddle)} super-node(s) straddle it. "
            f"A must-link across the barrier asserts one identity on both sides -- delete those "
            f"pinned_chains (or cut the tracklet at the crossing). First few: "
            + "; ".join(f"members {m} span {axis}=[{lo:.0f}, {hi:.0f}]" for _i, m, lo, hi in straddle[:5]))
    nl = sum(1 for s in sides if s == 0); nr = sum(1 for s in sides if s == 1)
    logger.info("Barrier CLOSED at %s=%g: %d node(s) left, %d right, %d unconstrained "
                "(no geometry). No identity may span it.", axis, pos, nl, nr, no_geom)
    return sides


def get_barrier_label_sides(animal_ids, cfg):
    """`barrier.left_animals` / `barrier.right_animals` -> {label: 0 left | 1 right}, fixing s[l].

    Without these the ILP chooses each label's side itself, which is safe but free to FLIP
    between solves: nothing distinguishes "animal 5 on the right" from "animal 5 on the left"
    except the evidence, so a marginal movie can come out mirrored. Declaring the sides pins the
    assignment you already verified, removes K binaries from the search, and -- because
    prototypes make the label the animal id -- carries across every movie of a session, since an
    animal does not change which side of a wall it lives on.

    Names ANIMALS, mapped through animal_ids like every other identity key, so
    prototypes_exclude offsets are handled. Naming one animal on both sides raises.
    """
    b = cfg.get("barrier") or {}
    if not b or str(b.get("state", "open")).lower() != "closed":
        return {}
    left = list(b.get("left_animals") or [])
    right = list(b.get("right_animals") or [])
    if not left and not right:
        return {}
    if animal_ids is None:
        raise ValueError("barrier.left_animals / right_animals require `prototypes_path`: "
                         "without fixed prototypes a label is an arbitrary per-movie index, so "
                         "naming an animal is meaningless")
    label_of = {int(a): i + 1 for i, a in enumerate(animal_ids)}
    out = {}
    for grp, val, key in ((left, 0, "left_animals"), (right, 1, "right_animals")):
        for a in grp:
            a = int(a)
            if a not in label_of:
                raise ValueError(f"barrier.{key} names animal {a}, which is not in this "
                                 f"prototype set {sorted(label_of)}")
            l = label_of[a]
            if l in out and out[l] != val:
                raise ValueError(f"barrier: animal {a} is named in BOTH left_animals and "
                                 f"right_animals")
            out[l] = val
    logger.info("Barrier: %d label side(s) fixed by config (%d left, %d right); the rest are "
                "chosen by the ILP.", len(out), sum(1 for v in out.values() if v == 0),
                sum(1 for v in out.values() if v == 1))
    return out


def get_closed_identities(nodes, animal_ids, cfg, forced):
    """`identity_closed: {8: [[0, 10000]]}` -> [(node, label)] pairs fixing x[i, l] = 0 on every
    node INSIDE the named window that is not forced to that label. Paired with
    `pinned_nodes_identity`, this is what FREEZES an identity over the frames you inspected.

    Why must-link alone cannot freeze a chain. `pinned_chains` contracts the chain into one
    super-node, fixing that its members share a label and (via pin_implies_keep) that the label
    is not 0. It says nothing about WHICH label, and nothing about who ELSE may take it. So on
    the next solve any unconstrained tracklet may be assigned the same identity, and the
    post-solve truncation then re-ranks the frames both supply -- the chain's contents change
    without a single one of its own labels having moved. Lock all only looks like a freeze
    because it is TOTAL (every node in range is kept+must-linked or in pinned_nodes_outlier),
    leaving no free node to wander in.

    Closing the identity supplies the missing half. `pinned_nodes_identity` says who IS in
    identity l (x[i,l] == 1); this says who is NOT (x[j,l] == 0). Neither alone is a freeze; the
    pair is, because between them every node in the window is decided with respect to label l,
    which is what "the contents of identity l" needs -- membership is a fact about the whole
    assignment, not about the chain's own nodes.

    THE WINDOW IS NOT OPTIONAL, and shipping this key without one was a real bug. A freeze is
    evidence about the frames a human actually inspected. Applied globally it also asserts the
    identity is ABSENT everywhere else: locking six chains that a 0-10000 inspection had grown
    to t~16000 forbade those six animals across all 71,992 frames, pushing 281,176 later
    detections into cluster 0 and leaving 4.5 of 12 identities present per frame. It made the
    ILP pathological too -- half the labels unavailable to ~20k nodes that plainly needed them,
    so the overlap repulsion had nowhere to put them, the relaxation collapsed and Gurobi's tree
    ate 64 GB and got SIGKILLed. Scoped to its window the constraint says only what was checked.

    A node counts as inside by the same `identity_span_overlap_frac` rule the identity_* windows
    use (shared `nodes_mostly_inside`), so a tracklet that merely overhangs the window keeps its
    freedom -- exactly the tracklets whose fate the inspection could not have settled.

    Requires fixed prototypes, and names ANIMALS, mapped through animal_ids like every other
    identity key (with prototypes_exclude [3,5,7], animal 12 is label 9).

    Closing a label nothing is forced to would forbid it on every node in the window and erase
    the identity there. That is a missing or typo'd pinned_nodes_identity far more often than an
    intended erasure, so it raises.
    """
    closed = cfg.get("identity_closed") or {}
    if not closed:
        return []
    if isinstance(closed, (list, tuple)):
        raise ValueError(
            f"identity_closed must be {{animal: [[t0, t1], ...]}}, not a bare list {list(closed)}. "
            f"A closure without a window silently asserts the animal is absent from the whole "
            f"movie; give each the frame range you actually inspected, e.g. "
            f"identity_closed: {{{list(closed)[0]}: [[0, 10000]]}}")
    if animal_ids is None:
        raise ValueError("identity_closed requires `prototypes_path`: without fixed prototypes "
                         "the ILP labels are arbitrary per-movie indices, so naming an animal is "
                         "meaningless")
    label_of = {int(a): i + 1 for i, a in enumerate(animal_ids)}
    frac_req = float(cfg.get("identity_span_overlap_frac", 0.5))

    owners_of = defaultdict(set)
    for i, l in forced:
        owners_of[int(l)].add(int(i))

    out = []
    for a, spans in closed.items():
        a = int(a)
        if a not in label_of:
            raise ValueError(f"identity_closed names animal {a}, which is not in this prototype "
                             f"set {sorted(label_of)} (excluded, or not in the file)")
        l = label_of[a]
        owners = owners_of.get(l, set())
        if not owners:
            raise ValueError(
                f"identity_closed names animal {a} (label {l}) but pinned_nodes_identity assigns "
                f"no tracklet to it, so closing it would forbid the label on every node in the "
                f"window and erase the identity there. Close an identity only alongside the "
                f"pinned_nodes_identity entries that say who is in it.")
        if not spans:
            raise ValueError(f"identity_closed for animal {a} has no window. Give the frame "
                             f"range you inspected, e.g. {{{a}: [[0, 10000]]}}")
        for span in spans:
            t0, t1 = int(span[0]), int(span[1])
            if t1 < t0:
                t0, t1 = t1, t0
            inside, partial = nodes_mostly_inside(nodes, t0, t1, frac_req)
            add = [i for i in inside if i not in owners]
            out.extend((i, l) for i in add)
            logger.info("identity_closed: animal %s (label %d) closed on [%d, %d] -> %d owner "
                        "super-node(s) keep it, %d node(s) at least %.0f%% inside are forbidden; "
                        "%d overlap less and stay free.",
                        a, l, t0, t1, len(owners), len(add), 100 * frac_req, partial)
            if not add:
                logger.warning("identity_closed: animal %s window [%d, %d] forbids the label on "
                               "NO node -- the freeze does nothing. Check the window covers the "
                               "frames you inspected.", a, t0, t1)
    return sorted(set(out))


def solve_labeling_ilp(n, K, unary, attract_edges, overlap_pairs, kept_mask, cfg,
                       cannot_link=(), forbidden=(), outlier_mask=None, forced=(), sides=None,
                       label_sides=None):
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

    # Barrier: no identity may span a closed physical divider. One binary per label picks the
    # side, so the ILP assigns animals to sides itself instead of the user declaring it. Only
    # labels >= 1 are constrained -- the outlier class is not an identity and has no location.
    if sides is not None:
        sv = m.addVars(K + 1, vtype=GRB.BINARY, name="side")
        for l, v in (label_sides or {}).items():
            m.addConstr(sv[l] == v, name=f"sidefix[a{l}]")
        n_bar = 0
        for i, sd in enumerate(sides):
            if sd is None:
                continue
            for l in range(1, K + 1):
                if sd == 0:
                    m.addConstr(x[i, l] + sv[l] <= 1, name=f"barrierL[{i},a{l}]")
                else:
                    m.addConstr(x[i, l] <= sv[l], name=f"barrierR[{i},a{l}]")
                n_bar += 1
        logger.info("Barrier: %d (node, label) constraint(s) + %d side binaries.", n_bar, K)

    # pinned_nodes_identity: this tracklet IS this animal. Hard, and it subsumes keep --
    # x[i,l] == 1 for l >= 1 already implies x[i,0] == 0.
    for (i, l) in forced:
        m.addConstr(x[i, l] == 1, name=f"ident_{i}_{l}")
    if forced:
        logger.info("pinned_nodes_identity: %d node(s) forced to a specific animal.", len(forced))

    # identity_start_frames / identity_end_frames: an animal outside its active window
    # cannot be carried. Unary variable fixings, so Gurobi presolves them away.
    for (i, l) in forbidden:
        m.addConstr(x[i, l] == 0, name=f"ended_{i}_{l}")
    if forbidden:
        logger.info("identity windows: %d (node, label) pair(s) forbidden.", len(forbidden))

    # cannot-link (Unlink / pinned_edges_false): ALWAYS hard. Unlike the overlap
    # repulsion -- a heuristic prior that strong appearance evidence may override --
    # this is a human assertion that the two tracklets are different identities.
    # A cannot-link between two nodes FORCED to the same animal is unsatisfiable. Catch it here
    # by name rather than letting it surface as a bare INFEASIBLE (the dual of the must-link
    # check in get_cannot_link_pairs, which only covers pinned_chains).
    forced_of = {int(i): int(l) for i, l in forced}
    clash = [(i, j, forced_of[i]) for (i, j) in cannot_link
             if forced_of.get(i) is not None and forced_of.get(i) == forced_of.get(j)]
    if clash:
        raise ValueError(
            "pinned_edges_false (unlink) contradicts pinned_nodes_identity: "
            + "; ".join(f"nodes {i} x {j} are both forced to animal-label {l}" for i, j, l in clash[:10])
            + ". An unlink asserts DIFFERENT identities while the assignment asserts the same one; "
              "delete one of the two.")
    for (i, j) in cannot_link:
        for l in range(1, K + 1):
            m.addConstr(x[i, l] + x[j, l] <= 1, name=f"cnl_{i}_{j}_{l}")
    if cannot_link:
        logger.info("Cannot-link (Unlink) pairs: %d -- hard must-not-link.", len(cannot_link))

    # keep -> non-outlier
    for i in range(n):
        if kept_mask[i]:
            m.addConstr(x[i, 0] == 0, name=f"keep_{i}")
        elif outlier_mask is not None and outlier_mask[i]:
            # pinned_nodes_outlier: judged an outlier, must stay one -- the exact dual of keep.
            # NOT a drop: a drop removes the node from the problem, so it vanishes from cluster 0
            # too. Forcing the label instead keeps it visible and promotable, and leaves its
            # motion/overlap edges in the graph so neighbouring decisions see the same costs as
            # the solve being reproduced.
            m.addConstr(x[i, 0] == 1, name=f"outlier_{i}")

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
        # Gurobi's INFEASIBLE says nothing about WHICH constraints clash, and hunting it by hand
        # cost three rounds on cohort7 _1959. The IIS is an irreducible conflicting subset --
        # typically 3 constraints -- and every constraint here is named after the tracklets and
        # animal involved, so it points straight at the pair to fix.
        detail = ""
        if m.Status == GRB.INFEASIBLE:
            try:
                logger.info("Model infeasible -- computing IIS to name the conflict...")
                m.computeIIS()
                names = [c_.ConstrName for c_ in m.getConstrs() if c_.IISConstr]
                if names:
                    detail = ("\n  Irreducible conflicting constraints (%d):\n    %s"
                              % (len(names), "\n    ".join(names[:40])))
                    logger.error("IIS: %d conflicting constraint(s): %s", len(names), names[:40])
            except gp.GurobiError as e:
                logger.warning("computeIIS failed: %s", e)
        raise RuntimeError(f"ILP found no feasible solution (status {m.Status})."
                           f"{detail}\n  Common cause: a cannot-link (pinned_edges_false) between "
                           f"two tracklets that pinned_nodes_identity forces to the SAME animal -- "
                           f"x[i,l]=1, x[j,l]=1 cannot satisfy x[i,l]+x[j,l]<=1. Delete the unlink, "
                           f"or the identity assignment.")
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        labels[i] = int(np.argmax([x[i, l].X for l in L]))
    return labels, m.ObjVal


def main(config_path):
    cfg = load_config(str(config_path))
    print_config(cfg)
    t0 = time.time()

    # The `mode` key is gone -- apt_faithful is the only formulation. A config that still
    # carries mode: correlation_clustering is now an error rather than a silent path change.
    if cfg.get("mode") not in (None, "apt_faithful"):
        raise ValueError(f"unsupported mode {cfg['mode']!r}: this script only does apt_faithful "
                         f"(the `mode` config key was removed; drop it)")
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
    # `prototypes_path`: reuse identity prototypes distilled from an ALREADY-FIXED movie of the
    # same recording session (extract_prototypes.py) instead of re-deriving them here. The
    # clustering keys (prototype_mode / num_animals / cluster_threshold / min_cluster_frac /
    # centre_* / close_thresh_percentile) are then inert, and -- the real payoff -- the ILP
    # labels are the SAME animals across every movie, so the output tracklet_id becomes a
    # session-wide animal id rather than a per-movie arbitrary index.
    proto_path = cfg.get("prototypes_path")
    if proto_path:
        num_tracks = None
        logger.info("Prototype mode: FIXED (from %s); clustering keys ignored.", proto_path)
    elif proto_mode is not None:
        assert proto_mode in ("thresh", "known"), f"bad prototype_mode {proto_mode!r} (thresh|known)"
        assert num_animals is not None, "prototype_mode set -> num_animals is required in the config"
        num_tracks = int(num_animals) if proto_mode == "known" else None
    else:
        num_tracks = cfg.get("num_tracks")
        proto_mode = "known" if num_tracks else "thresh"
    if not proto_path:
        logger.info("Prototype mode: %s (num_animals=%s).", proto_mode, num_animals)
    n_ex = cfg.get("cluster_n_samples", 25)
    seed = cfg.get("seed", 0)
    solution_csv = Path(cfg["solution_csv"]).expanduser()
    solution_csv.parent.mkdir(parents=True, exist_ok=True)

    if "outlier_as_singleton" in cfg:
        logger.warning("`outlier_as_singleton` is retired and ignored: outliers now always go to "
                       "CLUSTER 0 (%s). Remove the key.",
                       outliers_path(cfg["solution_csv"]).name)
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
        tids, E, frames = load_all_tracklets(csv_path, emb_path, emb_prefix,
                                             t_min, t_max, n_ex, rng)
    geom = load_geometry(csv_path, num_keypoints, t_min, t_max)
    pin_keep = cfg.get("pin_implies_keep", True)
    nodes = build_super_nodes(tids, E, frames, geom, dropped, kept, pinned_chains, n_ex, rng,
                              pin_implies_keep=pin_keep)
    n = len(nodes)
    n_emb = sum(nd["embs"] is not None for nd in nodes)
    n_dropped = len(dropped & set(tids))          # only drops that name an in-window tracklet
    logger.info("%d tracklets -> %d after drop -> %d super-nodes (%d pinned chains); "
                "%d node(s) with embeddings, %d without (motion-only).",
                len(tids), len(tids) - n_dropped, n, len(pinned_chains), n_emb, n - n_emb)

    # A pin naming a tracklet that is not in the node set can never be honoured -- it just
    # vanishes inside mustlink_groups' intersection. Say so rather than failing silently.
    known = set(tids)
    orphan = sorted({int(t) for chain in pinned_chains for t in chain} - known)
    if orphan:
        logger.warning("%d pinned tracklet(s) absent from the input CSV and therefore ignored: "
                       "%s%s", len(orphan), orphan[:20], " ..." if len(orphan) > 20 else "")
    if pin_keep:
        logger.info("pin_implies_keep=True: pinned tracklets are forced non-outlier "
                    "(%d kept node(s) total).", int(sum(nd["kept"] for nd in nodes)))

    animal_ids = None                 # set below iff prototypes came from a file
    cannot_link = get_cannot_link_pairs(nodes, cfg.get("pinned_edges_false"))
    if cannot_link:
        logger.info("%d unlink pair(s) from pinned_edges_false -> cannot-link.", len(cannot_link))

    overlap_pairs = get_overlap_pairs(nodes, cfg) if cfg.get("enforce_overlap", True) else []
    logger.info("Temporally-overlapping pairs (>%.0f%% of shorter node): %d -- repulsed either "
                "softly or hard, see the next overlap line.",
                100 * cfg["overlap_frac_thresh"], len(overlap_pairs))

    # 2) build the ILP terms
    apt_npz = cfg.get("apt_unary_npz")
    if apt_npz and Path(apt_npz).exists():
        # APT's OWN cluster centers + unary (get_id_cluster_centers / get_graph_cut_unary),
        # precomputed to npz. unary_vals is per trk target; map to super-nodes (mean over
        # members) and scale by unary_wt exactly as APT (k_way_graph_cut(unary_vals*unary_wt)).
        z = np.load(apt_npz)
        uv, K = z["unary_vals"], int(z["K"])
        unary = np.array([uv[nd["members"]].mean(axis=0) for nd in nodes]) * cfg["unary_wt"]
        logger.info("apt_faithful: APT unary from %s (K=%d).", apt_npz, K)
    elif proto_path:
        centres, K, animal_ids = load_fixed_prototypes(proto_path, nodes, cfg)
        unary = get_unary(nodes, centres, cfg) * cfg["unary_wt"]
    else:
        # Prototypes come from the EMBEDDED nodes only -- they are cluster centres in
        # id-embedding space, so a node with no embedding has nothing to contribute and
        # would crash the np.stack. (It is excluded by centre_min_len > 50 anyway; the
        # embedding-less ones are all <= 5 frames.)
        emb_idx = [i for i, nd in enumerate(nodes) if nd["embs"] is not None]
        assert emb_idx, "no tracklet has embeddings -- cannot build prototypes"
        emb_nodes = [nodes[i] for i in emb_idx]
        dist_diag = intra_spread([nd["embs"] for nd in emb_nodes])
        oa_tid = spatial_overlap(csv_path, num_keypoints, t_min, t_max)   # APT get_overlap_value
        oa = np.array([np.mean([oa_tid.get(t, 0.0) for t in nd["members"]]) for nd in nodes])
        logger.info("Spatial overlap oa: median=%.3f, frac<%.2f=%.2f",
                    float(np.median(oa)), cfg["centre_max_overlap"],
                    float(np.mean(oa < cfg["centre_max_overlap"])))
        centres = get_prototypes(emb_nodes, dist_diag, oa[emb_idx], num_tracks, cfg)
        K = len(centres)
        logger.info("apt_faithful: %d prototype identities (from %d embedded node(s)).",
                    K, len(emb_idx))
        # Both cost terms are always on now (the use_id_embedding_distance /
        # use_spatial_distance switches were removed -- there is no supported reason to run
        # with either off). Appearance unary from the embedded nodes; motion pairwise below.
        unary = get_unary(nodes, centres, cfg) * cfg["unary_wt"]
    attract = get_motion_pairwise(nodes, cfg)
    logger.info("Attraction edges: %d", len(attract))

    # 3) solve
    kept_mask = np.array([nd["kept"] for nd in nodes])

    # pinned_nodes_outlier: freeze a previous solve's outlier verdict. Two contradictions must be
    # caught here rather than handed to Gurobi as an unexplained INFEASIBLE.
    pin_out = {int(t) for t in (cfg.get("pinned_nodes_outlier") or [])}
    outlier_mask = np.zeros(len(nodes), bool)
    if pin_out:
        both_keep, in_chain = [], []
        for i, nd in enumerate(nodes):
            hit = [t for t in nd["members"] if t in pin_out]
            if not hit:
                continue
            if kept_mask[i]:
                # x[i,0] cannot be 0 and 1. `kept` also covers pin_implies_keep, so this catches
                # a tracklet that is must-linked into a chain (its super-node is forced in).
                (in_chain if len(nd["members"]) > 1 else both_keep).extend(hit)
                continue
            if len(nd["members"]) > 1:
                in_chain.extend(hit)         # sharing a super-node drags the whole chain out
                continue
            outlier_mask[i] = True
        if both_keep:
            raise ValueError(f"pinned_nodes_outlier and pinned_nodes_true both name tracklet(s) "
                             f"{sorted(set(both_keep))[:20]} -- x[i,0] cannot be 0 and 1")
        if in_chain:
            raise ValueError(f"pinned_nodes_outlier names tracklet(s) {sorted(set(in_chain))[:20]} "
                             f"that pinned_chains must-links into a chain: they share a super-node, "
                             f"so forcing them outlier would drag the whole identity out")
        logger.info("pinned_nodes_outlier: %d node(s) frozen as outliers (of %d named).",
                    int(outlier_mask.sum()), len(pin_out))

    forced = get_forced_identities(nodes, animal_ids, cfg)
    # x[i,l] == 1 for l >= 1 and x[i,0] == 1 cannot both hold. pin_out's own checks above only
    # see kept_mask and chain membership, so a bare assigned-and-frozen tracklet would reach
    # Gurobi as a bare INFEASIBLE. Reachable in one gesture now that Lock # writes
    # pinned_nodes_identity while Lock all writes pinned_nodes_outlier.
    if pin_out and forced:
        clash = sorted({t for i, _l in forced for t in nodes[i]["members"] if t in pin_out})
        if clash:
            raise ValueError(f"pinned_nodes_identity and pinned_nodes_outlier both name "
                             f"tracklet(s) {clash[:20]} -- a node cannot be an animal and the "
                             f"outlier class at once")
    forbidden = get_identity_window_forbidden(nodes, animal_ids, cfg)
    # A forced identity beats a window that forbids it -- the explicit assignment is the later,
    # more specific statement -- so drop any window entry that would contradict it.
    if forced:
        fset = set(forced)
        n0 = len(forbidden)
        forbidden = [p for p in forbidden if p not in fset]
        if len(forbidden) != n0:
            logger.warning("%d identity-window entr(ies) contradicted an explicit "
                           "pinned_nodes_identity and were dropped.", n0 - len(forbidden))
    # identity_closed is computed FROM `forced` and exempts exactly its owners, so it can never
    # contradict one -- it is unioned in after the window reconciliation above, not subjected to
    # it.
    forbidden = sorted(set(forbidden) | set(get_closed_identities(nodes, animal_ids, cfg, forced)))
    sides = get_barrier_sides(nodes, geom, cfg)
    label_sides = get_barrier_label_sides(animal_ids, cfg)
    labels, obj = solve_labeling_ilp(n, K, unary, attract, overlap_pairs, kept_mask, cfg,
                                     cannot_link=cannot_link, forbidden=forbidden,
                                     outlier_mask=outlier_mask, forced=forced, sides=sides,
                                     label_sides=label_sides)
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
        write_solution_truncated(nodes, labels, attract, csv_path, solution_csv, group, cfg,
                                 animal_ids=animal_ids,
                                 raw_rank=raw_tracklet_nodes(tids, frames, geom))
        logger.info("Done in %.1fs (truncate resolution).", time.time() - t0)
        return

    if cfg.get("resolve_overlaps", True):
        labels = resolve_cluster_overlaps(nodes, labels, attract, cfg)
        logger.info("After overlap resolution: %d non-outlier identities used.",
                    len({l for l in labels if l >= 1}))

    # 4) labels -> per-identity chains -> CSV. Outlier (0) tracklets become singletons if asked.
    to_animal, _first_outlier = label_mapper(animal_ids)
    tid_to_cluster, outlier_tids = {}, set()
    n_outlier_nodes = 0
    for nd, lab in zip(nodes, labels):
        if lab == 0:                                   # outlier -> cluster 0, not the solution
            n_outlier_nodes += 1
            outlier_tids.update(int(t) for t in nd["members"])
        else:
            for tid in nd["members"]:
                tid_to_cluster[tid] = to_animal(lab)

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

    # One detection per identity per frame, as APT's Tracklet.apply_ids leaves it: the overlaps
    # of <= overlap_resolve_frac that resolve_cluster_overlaps keeps are written tracklet by
    # tracklet in index order, so in a shared frame the higher raw tracklet overwrites the lower.
    sol = pd.read_csv(solution_csv, sep=r"\s+")
    src = "original_tracklet_id" if "original_tracklet_id" in sol.columns else "tracklet_id"
    n_before = len(sol)
    sol = sol.sort_values(["tracklet_id", "t", src]).drop_duplicates(["tracklet_id", "t"], keep="last")
    if len(sol) < n_before:
        logger.info("Drop resolution: %d residual same-frame detection(s) within an identity "
                    "removed (higher raw tracklet kept, as APT apply_ids).", n_before - len(sol))

    # APT delete_short on the written solution, and parent_id recomputed per chain (always: the
    # de-duplication above can remove a chain link).
    n_identities = len(by_cluster)
    maxframes = int(cfg.get("delete_short_maxframes", 10))
    sol2, n_ch, _ = delete_short_chains(sol, maxframes, log_tag="drop")
    short_ids = set(sol["id"]) - set(sol2["id"]) if n_ch else set()
    sol2.to_csv(solution_csv, sep=" ", index=False)
    n_identities -= n_ch

    raw = clip_window(pd.read_csv(csv_path, sep=r"\s+"), cfg)
    c0 = raw[raw["tracklet_id"].astype(int).isin(outlier_tids) | raw["id"].isin(short_ids)]
    write_cluster0(c0, csv_path, solution_csv, group, n_outlier_nodes, n_ch)

    logger.info("Done in %.1fs: %d tracklets -> %d identities -> %s",
                time.time() - t0, len(tid_to_cluster), n_identities, solution_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="path to configs_graph_cut.yaml")
    args = parser.parse_args()
    main(args.config)
