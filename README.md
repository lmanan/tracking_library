## Installation

### 1. Clone the Repository
First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/lmanan/tracking_library.git
cd tracking_library
```

### 2. Set Up the Conda Environment
Create and activate a dedicated Conda environment named `motile`:

```bash
conda create -n motile python==3.10
conda activate motile
```

### 3. Install Dependencies
Install the required packages, including `ilpy`, `scip`, and all dependencies:

```bash
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
```

### 4. Install tracking_library in Editable Mode
Finally, install the tracking_library repository in editable mode (useful for development):

```bash
pip install -e .
```
Full Setup Summary:

```bash
git clone https://github.com/lmanan/tracking_library.git
cd tracking_library
conda create -n motile python=3.10
conda activate motile
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
pip install -e .
```


## Getting Started

```bash
cd tracking_library/tracklet_stitching/manual
python infer_node_edge_selection.py
```

## What `infer_node_edge_selection.py` does

This script stitches short tracklets into full tracks using an Integer Linear Program (ILP).

**High-level steps:**

1. **Load config** — reads `configs/cohort_1vs2.yaml`, which specifies paths, cost attributes, and solver parameters.
2. **Build candidate graph** — each tracklet becomes a node; candidate edges connect tracklets that are close in space and time (controlled by `num_neighbors`, `max_spatial_distance`, `max_time_gap`).
3. **Compute statistics** — mean and std of each node/edge attribute are computed to normalise costs.
4. **Pin nodes/edges** (optional) — specific tracklets can be forced into or out of the solution before solving.
5. **Set up ILP** — costs and constraints (e.g. exact number of tracks, max one child per tracklet) are added to the `motile` solver.
6. **Solve** — the ILP selects which nodes and edges to include so that the total cost is minimised.
7. **Save solution** — the selected tracklets and links are written to `solution/solution_centroid.csv`.

## Data files

Sample data is already available under `tracking_library/tracklet_stitching/data/`.

| File | Description |
|------|-------------|
| `tracklet_keypoints.csv` | Spatial locations of keypoints for each tracklet at every time point. Columns: `tracklet_id`, `t`, and `(y, x)` coordinates for each keypoint (`kp0_y`, `kp0_x`, …). Used to compute pairwise keypoint distances between candidate tracklet pairs. |
| `tracklet_id_embeddings.csv` | Re-identification embedding vector for each tracklet at every time point. Columns: `tracklet_id`, `k` (frame index within the tracklet), and 32 embedding dimensions. The embedding captures appearance/identity; the distance between two tracklets' embeddings (`id_distance`) is used as an edge cost in the ILP. |

