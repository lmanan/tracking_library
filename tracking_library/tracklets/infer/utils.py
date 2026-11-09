"""Helpers for infer_graph_cut.py: reading tracklets, grouping them into nodes, and writing
the solution. The model itself (prototypes, costs, constraints, the ILP) is in
infer_graph_cut.py.
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def id_distance_matrix(E):
    """E:[n,n_ex,D] -> [n,n] median pairwise L2 (APT-style id distance)."""
    n = E.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(E[i][None, :, None, :] - E[:, None, :, :], axis=-1)
        D[i] = np.median(d, axis=(1, 2))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def mustlink_groups(pinned_chains, present):
    """Union-find over pinned_chains -> list of sets of tracklet ids (must-link groups)."""
    parent = {t: t for t in present}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for chain in pinned_chains:
        members = [int(t) for t in chain if int(t) in parent]
        for a, b in zip(members[:-1], members[1:]):
            parent[find(a)] = find(b)
    groups = defaultdict(set)
    for t in present:
        groups[find(t)].add(t)
    return list(groups.values())


# --------------------------------------------------------------------------------------
# Inputs: EVERY tracklet becomes a node; embeddings are optional.
# --------------------------------------------------------------------------------------
def load_all_tracklets(csv_path, embeddings_path, emb_prefix, t_min, t_max, n_ex, rng):
    """All tracklets from the keypoints CSV, with embeddings attached where they exist.

    Upstream only computes id embeddings for tracklets of >= 6 frames (APT's embedding
    eligibility). Those long tracklets carry the appearance UNARY; every tracklet, however
    short, still participates in the pairwise MOTION cost and appears in the output -- so
    the node set must be the keypoints CSV, not the embedding file.

    This is why infer_clusters.load_tracklets is not used: its inner merge on detection id
    deletes embedding-less tracklets outright. On cohort7_20251216_1659 that silently
    dropped 12,314 of 14,729 tracklets (84% of ids, 3.5% of detections -- all fragments of
    <= 5 frames), and any pinned_chains naming one of them degenerated with no warning
    (mustlink_groups intersects with the surviving tids).

    Returns (tids, E, frames) with tids sorted; E is a LIST aligned with tids whose entry is
    either a [n_ex, D] float32 sample of that tracklet's per-detection embeddings or None
    (no embedded detection); frames[i] is the tracklet's frame array. The list-of-arrays
    form keeps build_super_nodes' positional indexing working for callers that still pass a
    dense ndarray (infer_pairwise_ilp).
    """
    kp = pd.read_csv(csv_path, sep=r"\s+", usecols=["id", "t", "tracklet_id"])
    if t_min is not None:
        kp = kp[kp["t"] >= t_min]
    if t_max is not None:
        kp = kp[kp["t"] <= t_max]

    em = pd.read_csv(embeddings_path, sep=r"\s+")
    emb_cols = [c for c in em.columns if c.startswith(emb_prefix + "_")]
    # left join: keep every detection, embeddings where available
    df = kp.merge(em[["id"] + emb_cols], on="id", how="left")
    has = df[emb_cols[0]].notna().to_numpy() if emb_cols else np.zeros(len(df), bool)
    E_all = df[emb_cols].to_numpy(np.float32) if emb_cols else np.zeros((len(df), 0), np.float32)

    tids, E, frames = [], [], []
    n_emb = n_partial = 0
    for tid, sub in df.groupby("tracklet_id", sort=True):
        rows = df.index.get_indexer(sub.index)
        tids.append(int(tid))
        frames.append(sub["t"].to_numpy())
        m = has[rows]
        if not m.any():
            E.append(None)
            continue
        embs = E_all[rows[m]]
        sel = rng.choice(len(embs), size=n_ex, replace=(len(embs) < n_ex))
        E.append(embs[sel])
        n_emb += 1
        n_partial += int(not m.all())

    n_short = len(tids) - n_emb
    logger.info("CSV inputs: %d tracklet(s) -- %d with embeddings (unary + motion), "
                "%d without (motion only, %.1f%%). %d partially embedded.",
                len(tids), n_emb, n_short, 100.0 * n_short / max(len(tids), 1), n_partial)
    if n_short:
        lens = [len(frames[i]) for i in range(len(tids)) if E[i] is None]
        logger.info("Embedding-less tracklets: lengths min=%d median=%d max=%d, %d detection(s).",
                    int(np.min(lens)), int(np.median(lens)), int(np.max(lens)), int(np.sum(lens)))
    return tids, E, frames


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
                              first_c=c[0], last_c=c[-1], first_v=fv, last_v=lv,
                              # full spatial extent, for the barrier partition: endpoints alone
                              # would miss a tracklet that crosses and comes back.
                              y_min=float(c[:, 0].min()), y_max=float(c[:, 0].max()),
                              x_min=float(c[:, 1].min()), x_max=float(c[:, 1].max()))
    return geom


# --------------------------------------------------------------------------------------
# Super-nodes: apply drop (remove) + pin (contract must-link groups). One entry per node.
# --------------------------------------------------------------------------------------
def build_super_nodes(tids, E, frames, geom, dropped, kept, pinned_chains, n_ex, rng,
                      pin_implies_keep=True):
    """Returns list of super-node dicts: members(tid list), embs[n_ex,D] or None,
    t_start, t_end, length, first_c/last_c/first_v/last_v (from earliest/latest member), kept.

    `embs` is None when NO member has embeddings (short tracklets, see load_all_tracklets):
    such a node contributes no unary and is labeled by the motion pairwise alone. A group
    mixing embedded and embedding-less members pools only the embedded ones.

    `pin_implies_keep` (config `pin_implies_keep`, default True): a member named in
    pinned_chains marks the node kept, i.e. forced non-outlier (x[i,0] == 0). Without this,
    `pinned_chains` in this solver is must-link ONLY -- it contracts the pair into one
    super-node but lets the ILP still label it outlier -- so a user pressing `p` in the
    interactive tracker would not actually get the two tracklets retained. (The motile
    sibling, infer_node_edge_selection.py, pins both endpoint nodes True as part of the
    chain, hence the same semantics here.)
    """
    pos = {t: i for i, t in enumerate(tids)}
    work = [t for t in tids if t not in dropped]
    groups = [sorted(g) for g in mustlink_groups(pinned_chains, set(work))]
    pinned_tids = {int(t) for chain in (pinned_chains or []) for t in chain} if pin_implies_keep else set()

    nodes = []
    for g in groups:
        embs_g = [E[pos[t]] for t in g if E[pos[t]] is not None]
        if embs_g:
            embs_all = np.concatenate(embs_g, axis=0)
            sel = rng.choice(len(embs_all), size=n_ex, replace=(len(embs_all) < n_ex))
            embs = embs_all[sel]
        else:
            embs = None
        fr = np.unique(np.concatenate([frames[pos[t]] for t in g]))
        gg = [t for t in g if t in geom]
        first = min(gg, key=lambda t: geom[t]["t_start"]) if gg else g[0]
        last = max(gg, key=lambda t: geom[t]["t_end"]) if gg else g[0]
        nodes.append(dict(
            members=g, embs=embs, frames=fr,
            t_start=int(geom[first]["t_start"]) if gg else int(fr.min()),
            t_end=int(geom[last]["t_end"]) if gg else int(fr.max()),
            length=int(len(fr)),
            first_c=geom[first]["first_c"] if gg else np.zeros(2),
            last_c=geom[last]["last_c"] if gg else np.zeros(2),
            first_v=geom[first]["first_v"] if gg else np.zeros(2),
            last_v=geom[last]["last_v"] if gg else np.zeros(2),
            kept=any(t in kept for t in g) or any(t in pinned_tids for t in g)))
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


def nodes_mostly_inside(nodes, t0, t1, frac_req):
    """(indices, n_partial) of nodes whose overlap with [t0, t1] is >= frac_req of their extent.

    Shared by identity_start_frames / identity_end_frames / identity_forbidden_spans (which
    forbid a label inside a region) and by identity_closed (which forbids a label inside the
    window it was locked in). One predicate so "inside the window" cannot mean two things.

    NOTE the extent is [t_start, t_end], so a CONTRACTED super-node counts the gaps it does not
    occupy. Measuring occupancy instead was tried and reverted on request (2026-09-07); it fixed
    a locked chain being deleted despite sitting entirely outside its forbidden span, but did not
    address the opposite failure, where inside-tracklets fused into a mostly-outside group escape
    the ban. See CLAUDE.md, "Identity windows".
    """
    out, partial = [], 0
    for i, nd in enumerate(nodes):
        lo, hi = max(nd["t_start"], t0), min(nd["t_end"], t1)
        ov = hi - lo + 1
        if ov <= 0:
            continue
        extent = nd["t_end"] - nd["t_start"] + 1
        if ov / extent >= frac_req:
            out.append(i)
        else:
            partial += 1
    return out, partial


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

    Per cluster: sort WORST-connected first (weakest in-cluster motion attraction; shorter span
    breaks ties) so they are dropped first and well-connected tracklets arrive later and find
    their frames free -- APT's current `np.lexsort((tlen, -link_scores))` (APT be923f99,
    2026-08-28; it was best-first before, which dropped the well-connected tracklet of an
    overlapping pair). As in APT's _best_group_link_cost, a tracklet spanning > 80% of the movie
    counts as perfectly connected (sorted last, never dropped first). Then greedily walk that
    order, dropping any node whose frames are still doubly occupied for > overlap_resolve_frac
    of its length. Dropped nodes get the outlier label 0 (APT's `no_group`, removed with
    keep_all_preds=False). Overlaps of <= 5% survive here; the drop path's writer then keeps one
    detection per identity per frame the way APT's Tracklet.apply_ids does (last writer wins).

    Approximation: APT scores connection by its own link costs to group peers; here by the ILP's
    motion attraction (top-k mutual links), the same ordering in spirit. Occupancy is over each
    node's actual frames (= APT's span for contiguous raw tracklets); a dropped node decrements
    occupancy only at its overlapping frames.
    """
    if not labels.size:
        return labels
    resolve_frac = float(cfg.get("overlap_resolve_frac", 0.05))
    labels = labels.copy()
    n_dropped = 0
    span = {i: nd["t_end"] - nd["t_start"] + 1 for i, nd in enumerate(nodes)}          # APT tlen
    movie_span = max(nd["t_end"] for nd in nodes) - min(nd["t_start"] for nd in nodes) + 1  # tlen_mov
    for lab in range(1, int(labels.max()) + 1):
        members = [i for i in range(len(nodes)) if labels[i] == lab]
        if len(members) <= 1:
            continue
        # best in-cluster attraction per node; -inf if isolated (APT: inf cost -> worst -> first);
        # +inf for a tracklet spanning > 80% of the movie (APT: cost 0 -> best -> last)
        link = {}
        for i in members:
            if span[i] / movie_span > 0.8:
                link[i] = np.inf
                continue
            best = -np.inf
            for j in members:
                if j != i:
                    a = attract.get((min(i, j), max(i, j)))
                    if a is not None and a > best:
                        best = a
            link[i] = best
        order = sorted(members, key=lambda i: (link[i], span[i]))        # worst-connected first
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


def delete_short_chains(out, maxframes, *, log_tag=""):
    """APT delete_short, on the FINAL solution DataFrame (identity = `tracklet_id`).

    Reproduces deepnet/link_trajectories.py:delete_short as applied in link_trklet_id AFTER
    group_graph_cut: any final chain occupying `<= maxframes` frames is deleted from the
    output (APT sets its id to -1; we drop its rows). Occupancy is the number of DISTINCT
    frames the chain covers -- the truncate/save paths already guarantee <=1 detection per
    frame per identity, so distinct-t count == APT's non-dummy frame count `nframes`. This
    is length-based and blind to outlier status, so it keeps a long unstitched singleton
    (e.g. a 2790-frame track) while removing 1-5 frame fragments -- exactly APT's behavior
    (its `keep_all_preds=False` default; `maxframes_delete=10`).

    parent_id is (re)computed per surviving chain regardless of `maxframes`, so this also
    serves as the truncate path's parent-linkage step. maxframes<=0 disables the deletion.
    Returns (out, n_chains_removed, n_dets_removed).
    """
    n_chains_removed = n_dets_removed = 0
    if maxframes and maxframes > 0 and len(out):
        occ = out.groupby("tracklet_id")["t"].nunique()
        short_ids = occ.index[occ <= maxframes]
        if len(short_ids):
            n0 = len(out)
            out = out[~out["tracklet_id"].isin(short_ids)].copy()
            n_chains_removed = int(len(short_ids))
            n_dets_removed = int(n0 - len(out))
            logger.info("delete_short%s: removed %d chain(s) occupying <= %d frame(s) "
                        "(%d detection(s) dropped); %d chain(s) remain.",
                        f" ({log_tag})" if log_tag else "", n_chains_removed, maxframes,
                        n_dets_removed, int(out["tracklet_id"].nunique()) if len(out) else 0)
    # (re)compute parent_id per chain, ordered by time (chain-first detection -> parent 0).
    out = out.sort_values(["tracklet_id", "t"]).reset_index(drop=True)
    if len(out):
        ids = out["id"].astype(np.int64).to_numpy()
        chains = out["tracklet_id"].astype(np.int64).to_numpy()
        parent = np.zeros(len(out), dtype=np.int64)
        if len(out) > 1:
            parent[1:] = np.where(chains[1:] == chains[:-1], ids[:-1], 0)
        out["parent_id"] = parent
    return out, n_chains_removed, n_dets_removed


def clip_window(df, cfg):
    """Restrict to the solve's [t_min, t_max]. The raw CSV is read unwindowed in the output paths,
    so an outlier tracklet extending past the window would otherwise contribute frames the solve
    never saw -- cluster 0 must describe THIS solve, not the whole movie."""
    t_min, t_max = cfg.get("t_min"), cfg.get("t_max")
    if t_min is not None:
        df = df[df["t"] >= int(t_min)]
    if t_max is not None:
        df = df[df["t"] <= int(t_max)]
    return df


def outliers_path(solution_csv):
    """Sibling file holding cluster 0: `{solution stem}_outliers.csv`."""
    q = Path(solution_csv)
    return q.with_name(q.stem + "_outliers.csv")


def write_cluster0(df_rows, csv_path, solution_csv, group, n_outlier_nodes, n_short_chains):
    """Cluster 0 -- everything the SOLVER left out of the solution, kept visible but separate.

    Two populations land here: tracklets the ILP labeled 0 (outlier), and chains delete_short
    removed. Both were evaluated as candidates and rejected by the solve; neither belongs to an
    identity. A **dropped** tracklet (pinned_nodes_false) is NOT here and must never be -- in this
    solver drops are removed from `work` in build_super_nodes *before* the ILP, so they can't be
    label 0 and can't reach this function. Dropping removes a node from the computation; being an
    outlier is a verdict of the computation. Keeping the two distinct is what makes the drop
    button mean one thing.

    Cluster 0 is NOT a chain: its tracklets overlap in time, so there is no parent_id linkage and
    no one-detection-per-frame guarantee. Everything is written with tracklet_id = 0 and
    parent_id = 0, and the tracker renders it as a scatter rather than a track.

    Writing an EMPTY file (header only) when nothing was left out is deliberate: it keeps the
    file's presence a statement about *this* solve rather than a stale leftover of an older one.
    """
    out_path = outliers_path(solution_csv)
    cols = ["group", "id", "t", "y", "x", "parent_id", "tracklet_id",
            "original_id", "original_tracklet_id"]
    if df_rows is None or not len(df_rows):
        pd.DataFrame(columns=cols).to_csv(out_path, sep=" ", index=False)
        logger.info("Cluster 0: empty -> %s", out_path)
        return out_path
    out = df_rows.copy()
    out["original_id"] = out["id"].astype(int)
    out["original_tracklet_id"] = out["tracklet_id"].astype(int)
    out["tracklet_id"] = 0
    out["parent_id"] = 0
    if group is not None:
        out["group"] = group
    out = out.sort_values(["original_tracklet_id", "t"]).reset_index(drop=True)
    out.to_csv(out_path, sep=" ", index=False)
    logger.info("Cluster 0: %d detection(s) from %d raw tracklet(s) "
                "(%d outlier node(s), %d delete_short chain(s)) -> %s",
                len(out), out["original_tracklet_id"].nunique(),
                n_outlier_nodes, n_short_chains, out_path)
    return out_path


def label_mapper(animal_ids):
    """ILP label (1..K, an INDEX into `centres`) -> the identity written to tracklet_id.

    With in-movie prototypes the two coincide, so animal_ids is None and this is identity. With
    a prototype FILE they diverge as soon as the file's ids are not 1..K in order -- which is
    exactly what prototypes_exclude produces (drop animals 3,5,7 from a 12-animal file and
    animal 12 sits at index 9). Writing the raw index there would silently renumber every
    animal and destroy the cross-movie identity the prototype file exists to provide.

    Returns (to_animal, first_outlier_label); outlier singletons are numbered above every real
    animal id, not above K, so they can never collide with an unused animal's label.
    """
    if animal_ids is None:
        return (lambda lab: int(lab)), None
    ids = np.asarray(animal_ids, dtype=int)
    return (lambda lab: int(ids[int(lab) - 1])), int(ids.max()) + 1


def raw_tracklet_nodes(tids, frames, geom):
    """One singleton node per surviving RAW tracklet, in the shape get_motion_pairwise expects.

    This is deliberately NOT the solved node list: truncation must rank raw tracklets using
    values that do not depend on how pins happened to group them. A super-node's endpoints and
    velocities come from its earliest/latest member, so attraction computed on the contracted
    graph shifts the moment anything is locked -- and then solve -> Lock all -> solve is not a
    fixed point. Computing it here, pre-contraction, makes the ranking identical in both solves.

    For an unpinned solve every super-node IS a single raw tracklet, so this reproduces exactly
    what the old super-node ranking computed.

    DROPPED tracklets are deliberately INCLUDED. They can never be ranked (they are in no
    cluster), but get_motion_pairwise derives d_norm from the median per-frame step over all
    nodes and then keeps only mutual top-k links, so excluding them would make every surviving
    attraction value depend on what was dropped -- and Lock all drops things. Ranking must be a
    function of the raw CSV and the config alone, or the round trip is not a fixed point.
    """
    order = [t for t in tids if t in geom]
    pos = {t: i for i, t in enumerate(order)}
    fr_of = {t: frames[i] for i, t in enumerate(tids)}
    nodes = [dict(members=[t], frames=fr_of[t], length=int(geom[t]["length"]),
                  t_start=int(geom[t]["t_start"]), t_end=int(geom[t]["t_end"]),
                  first_c=geom[t]["first_c"], last_c=geom[t]["last_c"],
                  first_v=geom[t]["first_v"], last_v=geom[t]["last_v"])
             for t in order]
    return nodes, pos


def write_solution_truncated(nodes, labels, attract, csv_path, solution_csv, group, cfg,
                             animal_ids=None, raw_rank=None):
    """Overlap resolution by TRUNCATION (frame-level) instead of wholesale tracklet drop.

    Per cluster, greedily fill frames best-tracklet-first (strongest in-cluster attraction,
    longer breaks ties, tracklet id last so the order is total); each tracklet keeps ONLY the
    frames not already occupied by a higher-priority tracklet in its cluster.

    The ranking runs over RAW TRACKLETS, not super-nodes. Ranking super-nodes made truncation a
    no-op inside a must-linked chain -- one member in the cluster, nothing to resolve, so both
    sides of an internal overlap survived and the identity got two detections at one frame. It
    also made the outcome depend on how pins grouped things. `raw_rank` supplies the
    pre-contraction (nodes, pos) from raw_tracklet_nodes so both are fixed. This guarantees <=1 detection
    per identity per frame while preserving EVERY non-conflicting detection -- only the duplicate
    (physically-impossible) frames are removed, so full frame coverage is retained (unlike the
    wholesale drop, which also discards a losing tracklet's unique frames).

    Writes the solution CSV directly (identity = tracklet_id column, as the scorer expects).
    Outliers (label 0) go to CLUSTER 0 (the `_outliers.csv` sibling), never into this file."""
    labels = np.asarray(labels)
    to_animal, first_outlier = label_mapper(animal_ids)
    by_cluster = defaultdict(list)
    outlier_nodes = []
    for i, lab in enumerate(labels):
        (by_cluster[int(lab)].append(i) if lab >= 1 else outlier_nodes.append(i))

    kept_frames = {}          # tid -> set(frames kept)
    tid_to_cluster = {}       # tid -> identity label
    n_trunc = 0
    raw_nodes, raw_pos = raw_rank if raw_rank else (None, None)
    # Priority is INTRINSIC to the raw tracklet: longer first, tracklet id to break ties.
    #
    # It used to be "strongest in-cluster motion attraction, length as tie-break". That cannot be
    # deterministic across a Lock all round trip, because the key is a max over OTHER MEMBERS OF
    # THE CLUSTER: locking changes cluster membership (a tracklet the ILP labeled L but truncation
    # reduced to zero frames never reaches the solution CSV, so Lock all cannot see it and it ends
    # up dropped), which silently re-ranks the survivors. Observed as 15 detections swapping owner
    # -- e.g. frames 5840-5852 of chain 6 moving between raw 3402 and raw 376.
    #
    # Length is invariant to contraction, to drops and to cluster membership, so solve and
    # re-solve rank identically. It is also the more defensible rule on the merits: a contested
    # frame is one where two tracklets in the SAME identity disagree about where the animal is
    # (median ~80 px apart on cohort7 _1459), and the longer tracklet carries more evidence.
    for lab, members in by_cluster.items():
        raw_tids = [int(t) for i in members for t in nodes[i]["members"]]
        def _rank(t):
            ia = raw_pos.get(t) if raw_pos else None
            return (-(int(raw_nodes[ia]["length"]) if ia is not None else 0), t)
        order = sorted(raw_tids, key=_rank)
        occupied = set()
        for tid in order:
            ia = raw_pos.get(tid) if raw_pos else None
            fr = set(int(f) for f in (raw_nodes[ia]["frames"] if ia is not None else ()))
            keep = fr - occupied
            n_trunc += len(fr) - len(keep)
            occupied |= keep
            kept_frames[tid] = kept_frames.get(tid, set()) | keep
            tid_to_cluster[tid] = to_animal(lab)

    # Outliers go to CLUSTER 0, never to per-tracklet singleton chains. A singleton chain was
    # indistinguishable from a real identity in every consumer (lineage columns, chain counts,
    # delete_short) even though the solver had explicitly declined to assign it one.
    outlier_tids = {int(t) for i in outlier_nodes for t in nodes[i]["members"]}

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

    # APT delete_short + parent_id recompute. delete_short_chains drops final chains that
    # occupy <= delete_short_maxframes frames (APT's post-graph-cut behavior) and recomputes
    # parent_id per surviving chain (the raw parent_id carried over from the detection CSV is
    # the ORIGINAL tracklet linkage and is stale after re-clustering).
    maxframes = int(cfg.get("delete_short_maxframes", 10))
    before = out
    out, n_short, _ = delete_short_chains(out, maxframes, log_tag="truncate")
    # A chain delete_short removed was also "left out by the solve", so it joins cluster 0
    # rather than vanishing -- otherwise those detections are silently unaccounted for.
    # NB match on detection id, not index: delete_short_chains reset_index(drop=True)s its result.
    short_rows = before[~before["id"].isin(set(out["id"]))] if n_short else before.iloc[:0]

    Path(solution_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(solution_csv, sep=" ", index=False)
    c0 = clip_window(df, cfg)
    c0 = c0[c0["tracklet_id"].astype(int).isin(outlier_tids)]
    if len(short_rows):
        c0 = pd.concat([c0, short_rows[df.columns.intersection(short_rows.columns)]],
                       ignore_index=True)
    write_cluster0(c0, csv_path, solution_csv, group, len(outlier_nodes), n_short)
    logger.info("Truncate resolution: removed %d duplicate-frame detections; wrote %d identities, "
                "%d detections -> %s", n_trunc, out["tracklet_id"].nunique(), len(out), solution_csv)
