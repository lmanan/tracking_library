import logging
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from x_utils.general_utils import load_config, print_config
from tracking_utils.create_tracklet_candidate_graph import (
    create_tracklet_candidate_graph,
)
from motile import TrackGraph
from motile.solver import Solver
from tracking_utils.add_costs import add_costs
from tracking_utils.add_constraints import add_constraints
from tracking_utils.compute_graph_statistics import (
    compute_edge_statistics,
    compute_node_statistics,
)
from tracking_utils.get_groundtruth_mask import get_groundtruth_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

config = load_config("../configs/cohort_1vs2.yaml")
print_config(config)

csv_path = config["csv_path"]
embeddings_path = config["embeddings_path"]
num_neighbors = config["num_neighbors"]
max_spatial_distance = config["max_spatial_distance"]
max_time_gap = config["max_time_gap"]
edge_attributes = config["edge_attributes"]
node_attributes = config["node_attributes"]
num_tracks = config["num_tracks"]
embeddings_attribute_prefix = config["embeddings_attribute_prefix"]
set_exact_track_count = config["set_exact_track_count"]
set_exact_active_tracklets_per_frame = config["set_exact_active_tracklets_per_frame"]
t_min = config.get("t_min", None)
t_max = config.get("t_max", None)
max_time_gap_past = config.get("max_time_gap_past", 0)

gt_chains = config.get("pinned_chains", [])
gt_nodes = config.get("pinned_nodes", [])

candidate_graph = create_tracklet_candidate_graph(
    tracklet_csv_path=csv_path,
    tracklet_embeddings_path=embeddings_path,
    num_neighbors=num_neighbors,
    max_spatial_distance=max_spatial_distance,
    max_time_gap=max_time_gap,
    embeddings_attribute_prefix=embeddings_attribute_prefix,
    t_min=t_min,
    t_max=t_max,
    max_time_gap_past=max_time_gap_past,
)

# -------------------------------------------------------------------------
# Assign GT labels (dense: every node and edge gets GT=0 or GT=1)
# -------------------------------------------------------------------------
gt_node_true = set(gt_nodes)
gt_edge_set = set()

for chain in gt_chains:
    for nid in chain:
        gt_node_true.add(nid)
    for u, v in zip(chain[:-1], chain[1:]):
        if candidate_graph.has_edge(u, v):
            gt_edge_set.add((u, v))

for nid in candidate_graph.nodes:
    candidate_graph.nodes[nid]["GT"] = 1 if nid in gt_node_true else 0

for u, v in candidate_graph.edges:
    candidate_graph.edges[u, v]["GT"] = 1 if (u, v) in gt_edge_set else 0

logger.info(
    "GT labels — nodes: %d positive / %d total, edges: %d positive / %d total",
    len(gt_node_true & set(candidate_graph.nodes)),
    candidate_graph.number_of_nodes(),
    len(gt_edge_set),
    candidate_graph.number_of_edges(),
)

# -------------------------------------------------------------------------
# Build solver (same setup as infer_ilp.py)
# -------------------------------------------------------------------------
edge_statistics = compute_edge_statistics(
    candidate_graph, attributes=edge_attributes, frame_attribute="t_start"
)
node_statistics = compute_node_statistics(
    candidate_graph, attributes=node_attributes, frame_attribute="t_start"
)

track_graph = TrackGraph(candidate_graph, frame_attribute="t_start")
solver = Solver(track_graph)

add_costs(
    solver,
    edge_attributes=edge_attributes,
    edge_statistics=edge_statistics,
    node_attributes=node_attributes,
    node_statistics=node_statistics,
)
add_constraints(
    solver,
    num_tracks=num_tracks,
    max_children=1,
    set_exact_track_count=set_exact_track_count,
    set_exact_active_tracklets_per_frame=set_exact_active_tracklets_per_frame,
    pin_attribute=None,
)

# -------------------------------------------------------------------------
# Extract features and GT labels via motile feature matrix
# -------------------------------------------------------------------------
# Zero-initialise weights so features are extracted without a solved state.
n_weights = len(
    [
        *[f"Node Selection {a}_weight" for a in node_attributes],
        *[f"Node Selection {a}_constant" for a in node_attributes],
        *[f"Edge Selection {a}_regular_weight" for a in edge_attributes],
        *[f"Edge Selection {a}_regular_constant" for a in edge_attributes],
        "Appear_weight",
        "Appear_constant",
        "Disappear_constant",
    ]
)
solver.weights.from_ndarray(np.zeros(n_weights))

groundtruth, mask = get_groundtruth_mask(solver, gt_attribute="GT", mode="dense")

features = solver.features.to_ndarray()
logger.info(
    "features shape: %s, groundtruth shape: %s, mask shape: %s",
    features.shape,
    groundtruth.shape,
    mask.shape,
)

mask = mask.astype(bool)
learnable_cols = [0, 1, 2, 4, 6, 8, 10, 12, 13]
X = features[mask][:, learnable_cols]
y = groundtruth[mask].astype(int)

logger.info(
    "Training MLP on %d samples (%d positive, %d negative), %d features.",
    len(y),
    y.sum(),
    (y == 0).sum(),
    X.shape[1],
)

# -------------------------------------------------------------------------
# Train MLP with sample weights for class balance
# -------------------------------------------------------------------------
classes = np.unique(y)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
sample_weight = np.array([class_weight_dict[yi] for yi in y], dtype=float)

logger.info(
    "Class distribution: %s",
    ", ".join(
        f"class {c}: count={(y == c).sum()}, weight={class_weight_dict[c]:.3f}"
        for c in classes
    ),
)

clf = Pipeline(
    [
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=42,
                verbose=True,
                n_iter_no_change=500,
            ),
        ),
    ]
)
clf.fit(X, y, mlp__sample_weight=sample_weight)

train_acc = clf.score(X, y)
logger.info("MLP train accuracy: %.4f", train_acc)

# -------------------------------------------------------------------------
# Save model
# -------------------------------------------------------------------------
model_dir = Path("../models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "mlp.joblib"
joblib.dump(clf, model_path)
logger.info("Saved MLP model to %s", model_path)

# Save learnable_cols to config so infer_mlp.py can load the same columns
config_path = Path("../configs/cohort_1vs2.yaml")
with open(config_path) as f:
    saved_config = yaml.safe_load(f)
saved_config["mlp_learnable_cols"] = learnable_cols
with open(config_path, "w") as f:
    yaml.dump(saved_config, f, default_flow_style=False, sort_keys=False)
logger.info("Saved mlp_learnable_cols to %s", config_path)
