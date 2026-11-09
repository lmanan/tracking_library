# Tracklet Stitching — Infer

## Graph-cut solver — prototype-count modes (`infer_graph_cut.py`)

The `apt_faithful` mode labels every super-node with one of `K` identity prototypes or the
**outlier** label. `K` is the number of prototype cluster centres derived from the clean, long
subset (`get_prototypes`). How `K` is chosen is now an **explicit** config choice.

### Config keys

| Key | Meaning |
|-----|---------|
| `prototype_mode` | `"thresh"` or `"known"` — explicit selector (was implicit via `num_tracks`). |
| `num_animals` | Expected animal count. Used by **both** modes (see below). |
| `cluster_threshold` | Distance cut for the hierarchical clustering (threshold mode). |
| `min_cluster_frac` | Minimum cluster size as a fraction of total tracklet-length (threshold mode). |
| `num_tracks` | **Legacy fallback only.** If `prototype_mode` is absent, `num_tracks: null` ⇒ thresh, `num_tracks: N` ⇒ known. Ignored when `prototype_mode` is set. |

`prototype_mode` set ⇒ `num_animals` is **required** (asserted at startup).

### `known` mode

`K = num_animals` exactly: cut the dendrogram with `fcluster(..., t=num_animals, criterion="maxclust")`
and keep **every** cluster (the `min_cluster_frac` filter is **skipped**). Gives exactly
`num_animals` prototypes, regardless of `cluster_threshold` (the threshold value is unused —
fixing the count *is* choosing the cut height).

### `thresh` mode

Cut at `cluster_threshold` (so the natural clustering decides the raw count), then apply the
`min_cluster_frac` filter — **but** if `num_animals` is set and the configured fraction admits
**fewer** than `num_animals` prototypes, **lower the fraction** just enough to admit at least
`num_animals` clusters (down to the size of the `num_animals`-th largest cluster). It **only
lowers, never raises**: if the configured fraction already yields ≥ `num_animals`, it is left
untouched (a no-op). This is *not* equivalent to `known` mode — it is a **floor**, so a rich
clustering can exceed `num_animals`, whereas `known` pins it exactly.

The **operating** `min_cluster_frac` (after any lowering) is logged on **every solve**:

```
threshold mode: operating min_cluster_frac=0.0097 (configured 0.050, num_animals=9) -> 9 prototype(s) from 16 raw cluster(s).
```

### Session prototypes — `prototypes_path` (reuse across movies)

`prototypes_path` points at an `.npz` written by `extract_prototypes.py` from an **already-fixed**
movie of the same recording session. When set, `get_prototypes` is skipped entirely (along with
`intra_spread` / `spatial_overlap`, which exist only to pick its "clean" subset), and every
clustering key above is **inert** — `prototype_mode`, `num_animals`, `cluster_threshold`,
`min_cluster_frac`, `centre_*`, `close_thresh_percentile`.

The payoff is not speed: **ILP label `l` is the same animal in every movie**, so the output
`tracklet_id` becomes a session-wide animal id instead of a per-movie arbitrary index. Measured on
cohort7 `_1459` (t=[0,10000], no pins), the partition is essentially what in-movie clustering
produces anyway — 8 of 9 labels correspond 1.0000, the ninth at 0.9977 — it just gets meaningful
names.

**You never declare how many animals are present.** An absent animal's label simply costs more
than the right one for every node, and nothing forces a label to be used, so it goes unused on its
own. The count of identities is **emergent** (`N + 1` outliers, `N <= K`), not `K`. On `_1459`,
animal 3 landed at exactly 0 frames unaided.

`prototypes_exclude: [3, 5, 7]` (optional) drops named animals up front. It removes the small
phantom chains a *weak* prototype can still attract — on `_1459` animals 5 and 7 held 254 and 383
frames, which on exclusion were reabsorbed into real identities (99.37% of detections kept their
animal). Use it sparingly: a wrong guess is **silent**, whereas leaving the prototype in makes
absence announce itself as a near-empty cluster. Solve once, read the per-animal frame mass, then
declare.

**Label vs index (`label_mapper`).** The ILP labels are *indices into `centres`*; the animal id is
`animal_ids[label-1]`. These coincide only while the file's ids are `1..K` in order — which
`prototypes_exclude` breaks immediately (drop 3,5,7 from 12 animals and animal 12 sits at index 9).
Both output paths therefore remap through `label_mapper` before writing `tracklet_id`, and outlier
singletons are numbered above `max(animal_ids)`, not above `K`, so they cannot collide with an
unused animal's label. With in-movie prototypes `animal_ids` is `None` and the mapping is identity.

**Drift alarm.** `outlier_unary` is an *absolute* threshold on mean sample-to-centre distance, so
prototypes from another movie only work if this movie's distances land in the same range. Every
solve logs `Prototype fit ... dist-to-nearest median/p90/p99/max; N% above outlier_unary`, and
warns if over half the embedded nodes exceed it — otherwise appearance drift would quietly send a
whole movie to the outlier label and emit an empty solution with no error.

**Pooling many solved movies at once — `pool_prototypes.py`.** Once every movie of a session has
been solved under one `prototypes_path`, its `tracklet_id` IS the animal id, so there is nothing to
match: `pool_prototypes.py <root> out.npz` pools detections per animal across every
`<root>/cohort*/solution/solution_keypoints.csv` + `tracklet_embeddings.csv` (movies missing either
file are skipped and listed) and writes the same `.npz` layout `extract_prototypes.py` does. It prints
a per-movie table of each movie's own chain-mean against the pooled centres and flags any chain whose
nearest pooled centre is another animal's — the signature of a swapped identity in one movie. Built
2026-09-11 on cohort7: all 17 movies, 9.1M detections, no
flags. Animals 3/5/7 (present only in the first three movies) have a within-movie spread of ~1.0
against 0.1–0.4 for the rest, in the GT movie too — an intrinsic property of those chains, not drift.

### The outlier label (the "+1")

The ILP labels run `0..K`: label `0` is the **outlier** ("matches none of the `K` prototypes"),
labels `1..K` are the identities. So `num_animals` prototypes give `num_animals + 1` ILP labels —
the `+1` is the outlier bin, **not** an extra identity. Every node is forced to take exactly one
label (`x.sum(i,"*")==1`); a node goes outlier when its nearest-prototype embedding distance
exceeds `outlier_unary`. Outlier tracklets become their own singleton identities if
`outlier_as_singleton` (default), else are dropped — so a run can emit **more** identities than
`num_animals`.

### Choosing the target (this movie, cohort7_20251216_1329, frames 0–10000)

Cluster sizes (frac of total length) at `cluster_threshold=1.0` fall as
`13.9, 13.3, 12.6, 12.4, 11.4, 10.1, 6.0, 4.3%` then a cliff to `<1%`. So the data shows ~**8**
clean identities (the 8th, at 4.3%, is exactly what the default `min_cluster_frac=0.05` drops).
`num_animals: 8` → frac 0.043; `num_animals: 9` → frac ~0.0097 (reaches into the sub-1% tail).

## Truncation ranks raw tracklets (`overlap_resolution: truncate`)

`write_solution_truncated` resolves an in-cluster overlap by ranking **raw tracklets**, not
super-nodes, using `raw_tracklet_nodes` (built pre-contraction from the keypoints CSV).

Two bugs made the old super-node ranking wrong once anything was pinned:

- **A must-linked chain became one cluster member, so truncation did nothing.** Every member
  received the union of the super-node's frames, and both sides of an internal overlap survived —
  two detections, one identity, one frame. Reproducible before the fix by pinning raw 261 and 303
  (2 frames of overlap) on cohort7 `_1459`: duplicates at exactly t=5253 and t=5254.
- **The outcome depended on how pins grouped things**, so `solve → Lock all → solve` was not a
  fixed point.

Priority is now **intrinsic**: longer tracklet first, `tracklet_id` to break ties. It used to be
"strongest in-cluster motion attraction, length as tie-break", which cannot be deterministic — the
key is a max over *other members of the cluster*, and locking changes cluster membership (a
tracklet the ILP labels `L` but truncation reduces to zero frames never reaches the solution CSV,
so Lock all cannot see it). That was worth 15 detections swapping owner, e.g. frames 5840–5852 of
chain 6 moving between raw 3402 and raw 376.

Length is also the more defensible rule: a contested frame is one where two tracklets in the
**same** identity disagree about where the animal is (median ~80 px apart), and the longer
tracklet carries more evidence.

`raw_tracklet_nodes` deliberately **includes dropped tracklets**. They can never be ranked (they
are in no cluster), but `get_motion_pairwise` derives `d_norm` from the median per-frame step over
all nodes and then keeps only mutual top-k links — so excluding them would make every surviving
attraction value depend on what was dropped, and Lock all drops things.

An unpinned solve is **byte-identical** to the old behavior (verified: 69010 detections, 0 lost,
0 gained, label agreement 1.0000) — for an unpinned solve every super-node already *is* a single
raw tracklet.

## Blind Visual Test

**Goal:** Evaluate stitching quality by presenting trajectory crops to an external person who does not know the solution, and comparing their identity map to the ground-truth answer.

### Script: `blind_visual_test.py`

**Parameters:**
- `t_start`: first time point of interest
- `t_end`: last time point of interest
- `crop_size`: side length (pixels) of the square crop centred on the centroid; default 128 (i.e. 128 × 128)

**Inputs:**
- `solution_centroids.csv` — columns `group id t y x parent_id tracklet_id`; used to look up the centroid (y, x) of each detection and to identify the 12 chains

**Output:** 2 figures + 1 answer file

---

### Figure 1 — forward crops

- Layout: 12 rows × 10 columns
- Each row = one chain (ordered consistently, e.g. by chain ID)
- Each column = one time step starting from `t_start`
- Each cell = a small image crop centred on the centroid (y, x) for that chain at that time

### Figure 2 — reverse crops (jumbled)

- Same layout: 12 rows × 10 columns
- Each row = one chain, but in **reverse time** starting from `t_end`
- Each column = one time step counting backwards from `t_end`
- The **row order is randomly shuffled** so the chain-to-row mapping differs from Figure 1
- The shuffle map (shuffled chain ID → original chain ID) is saved as the **answer file**

---

### Test protocol

1. Show Figure 1 and Figure 2 to an external person
2. Ask them to match each row in Figure 2 to the corresponding row in Figure 1, based on the animal's appearance across its trajectory
3. Compare their proposed map to the saved answer file

---

## Web App

### Local development (your own machine)

```bash
# Terminal 1 — backend (from infer/ directory)
pip install flask
python server.py

# Terminal 2 — frontend dev server with hot reload
cd blind_test_app
npm install
npm run dev
# open http://localhost:5173
```

### Sharing on internal network (e.g. funkelab-ws4)

**One-time setup on the server machine:**

```bash
ssh lalitm@funkelab-ws4
cd /path/to/tracking_library/tracking_library/tracklet_stitching/infer

# build the frontend (re-run after any UI change)
cd blind_test_app && npm install && npm run build && cd ..
```

**Start the server in a persistent tmux session:**

```bash
tmux new -s blindtest
python server.py
# Ctrl+B, D  — detach (server keeps running after SSH disconnect)
```

**Share this URL with colleagues on the same WiFi:**

```
http://funkelab-ws4:5001
```

If the hostname doesn't resolve, find the IP first:

```bash
hostname -I | awk '{print $1}'   # e.g. 192.168.1.42
# → http://192.168.1.42:5001
```

**To stop the server:**

```bash
tmux attach -t blindtest   # reattach
# Ctrl+C to kill, then: tmux kill-session -t blindtest
```

**Notes:**
- The frame source path in `blind_visual_test_configs.yaml` must be accessible from the server machine. By default it reads `video_path` (e.g. `movie.avi`); comment that out to fall back to the zarr array (`zarr_path` + `group`). Update the path if it differs from your local Mac.
- Rebuild the frontend (`npm run build`) after any change to `blind_test_app/src/`.
- Flask serves both the static frontend and the `/api/refresh` endpoint from a single process on port 5001.

---

## Web App — UI Notes

- **Score view (answer key):** rows are indexed by Fig1 letter (left panel). Each row shows `Fig1 A = Fig2 X` — left panel always drives the ordering.
- **"Show solution" button:** reorders the right panel (Fig2) to match the left panel row-for-row using the answer file. Toggles to "Hide solution" to restore the original shuffled order.
