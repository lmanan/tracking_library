import logging
import time
import numpy as np
import yaml
from pathlib import Path

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
from ssvm.soft_margin_loss import SoftMarginLoss
from ssvm.bundle_method import BundleMethod

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
gt_nodes_false = config.get("pinned_nodes_false", [])

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
gt_node_false = set(gt_nodes_false)
gt_edge_set = set()

for chain in gt_chains:
    for nid in chain:
        gt_node_true.add(nid)
    for u, v in zip(chain[:-1], chain[1:]):
        if candidate_graph.has_edge(u, v):
            gt_edge_set.add((u, v))

for nid in candidate_graph.nodes:
    if nid in gt_node_true:
        candidate_graph.nodes[nid]["GT"] = 1
    else:
        candidate_graph.nodes[nid]["GT"] = 0  # includes pinned_nodes_false explicitly

for u, v in candidate_graph.edges:
    candidate_graph.edges[u, v]["GT"] = 1 if (u, v) in gt_edge_set else 0

logger.info(
    "GT labels — nodes: %d positive / %d negative (explicit) / %d total, edges: %d positive / %d total",
    len(gt_node_true & set(candidate_graph.nodes)),
    len(gt_node_false & set(candidate_graph.nodes)),
    candidate_graph.number_of_nodes(),
    len(gt_edge_set),
    candidate_graph.number_of_edges(),
)

# -------------------------------------------------------------------------
# Build solver
# -------------------------------------------------------------------------
edge_statistics = compute_edge_statistics(
    candidate_graph, attributes=edge_attributes, frame_attribute="t_start"
)
for attr, stats in edge_statistics.items():
    mean, std = stats["regular"]
    logger.info("Edge statistic [%s]: mean=%.4f, std=%.4f", attr, mean, std)

node_statistics = compute_node_statistics(
    candidate_graph, attributes=node_attributes, frame_attribute="t_start"
)
for attr, (mean, std) in node_statistics.items():
    logger.info("Node statistic [%s]: mean=%.4f, std=%.4f", attr, mean, std)

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
# Extract features and GT labels
# -------------------------------------------------------------------------
groundtruth, mask = get_groundtruth_mask(solver, gt_attribute="GT", mode="dense")

features = solver.features.to_ndarray()
logger.info(
    "features shape: %s, groundtruth shape: %s, mask shape: %s",
    features.shape,
    groundtruth.shape,
    mask.shape,
)

mask = mask.astype(bool)

# Zero out features outside the mask and outside learnable columns.
# learn_features is (d, N) as expected by SoftMarginLoss.
learnable_cols = [0, 1, 2, 4, 6, 8, 10, 12, 13, 15, 16]

features[~mask] = 0.0
learn_features = features.T.copy()  # (d, N)
learn_mask = np.zeros(learn_features.shape[0], dtype=bool)
learn_mask[learnable_cols] = True
learn_features[~learn_mask] = 0.0

logger.info(
    "learn_features shape: %s, learnable dims: %s", learn_features.shape, learnable_cols
)

# -------------------------------------------------------------------------
# SSVM: soft margin loss + bundle method optimisation
# -------------------------------------------------------------------------
soft_margin_loss = SoftMarginLoss(
    constraints=solver.constraints,
    features=learn_features,
    ground_truth=groundtruth,
    solver=solver,
    mask=mask,
)

bundle = BundleMethod(
    value_gradient_callback=soft_margin_loss.value_and_gradient,
    dims=learn_features.shape[0],
    regularizer_weight=1e-5,
    eps=1e-6,
)

t0 = time.time()
w = bundle.optimize(max_iterations=100)
logger.info("SSVM optimisation took %.2fs.", time.time() - t0)
logger.info("w = %s", w)

# -------------------------------------------------------------------------
# Map learned weights back to weights_data
# -------------------------------------------------------------------------
# w is full-dimensional (non-learnable dims are zero); read off directly.
weights_data = {}

for attr in node_attributes:
    weights_data[f"Node Selection {attr}_weight"] = 0.0
    weights_data[f"Node Selection {attr}_constant"] = 0.0

for attr in edge_attributes:
    weights_data[f"Edge Selection {attr}_regular_weight"] = 0.0
    weights_data[f"Edge Selection {attr}_regular_constant"] = 0.0

weights_data["Appear_weight"] = 0.0
weights_data["Appear_constant"] = 0.0
weights_data["Disappear_constant"] = 0.0

for i, key in enumerate(weights_data):
    weights_data[key] = float(w[i])

logger.info("Learned weights_data: %s", weights_data)

# -------------------------------------------------------------------------
# Save learned weights to config
# -------------------------------------------------------------------------
config_path = Path("../configs/cohort_1vs2.yaml")
with open(config_path) as f:
    saved_config = yaml.safe_load(f)

saved_config["weights_data"] = {k: float(v) for k, v in weights_data.items()}

with open(config_path, "w") as f:
    yaml.dump(saved_config, f, default_flow_style=False, sort_keys=False)

logger.info("Saved learned weights to %s", config_path)
