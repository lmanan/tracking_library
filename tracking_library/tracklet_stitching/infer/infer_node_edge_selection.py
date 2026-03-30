import logging
from pathlib import Path
import numpy as np
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
from tracking_utils.get_solution_graph import get_solution_graph
from tracking_utils.compute_solution_statistics import (
    log_average_occupancy,
    log_solution_tracks,
    log_edge_margin_ranking,
)
from data_utils.save_tracklet_csv_data import save_tracklet_csv_data

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
# Pin specified nodes and edges before building the TrackGraph.
# pinned_nodes:       tracklet IDs to force INTO the solution (pin=True).
# pinned_nodes_false: tracklet IDs to force OUT of the solution (pin=False).
# pinned_chains:      ordered tracklet-ID sequences; all nodes and the
#                     edges connecting consecutive pairs are pinned True.
pinned_nodes = config.get("pinned_nodes", [])
pinned_nodes_false = config.get("pinned_nodes_false", [])
pinned_chains = config.get("pinned_chains", [])


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


for nid in pinned_nodes:
    if nid in candidate_graph.nodes:
        candidate_graph.nodes[nid]["pin"] = True

for nid in pinned_nodes_false:
    if nid in candidate_graph.nodes:
        candidate_graph.nodes[nid]["pin"] = False

for chain in pinned_chains:
    for nid in chain:
        if nid in candidate_graph.nodes:
            candidate_graph.nodes[nid]["pin"] = True
    for u, v in zip(chain[:-1], chain[1:]):
        if candidate_graph.has_edge(u, v):
            candidate_graph.edges[u, v]["pin"] = True

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
    pin_attribute="pin",
)

if "weights_data" in config:
    weights_data = config["weights_data"]
    logger.info("Using learned weights from config.")
else:
    weights_data = {}
    weights_data["Node Selection tracklet_length_weight"] = -1
    weights_data["Node Selection tracklet_length_constant"] = 0

    for attr in edge_attributes:
        weights_data[f"Edge Selection {attr}_regular_weight"] = 1
        weights_data[f"Edge Selection {attr}_regular_constant"] = 0

    # Add Appear and Disappear costs
    weights_data["Appear_weight"] = 0
    weights_data["Appear_constant"] = 1
    weights_data["Disappear_constant"] = 1

weights_array = np.array(list(weights_data.values()))
logger.info(f"Using weights: {weights_array}")
solver.weights.from_ndarray(weights_array)
logger.info("Solving ILP optimization...")
solution = solver.solve(verbose=False)
solution_graph = get_solution_graph(solver, solution)
logger.info(
    f"# solution nodes: {solution_graph.number_of_nodes()}, # solution edges: {solution_graph.number_of_edges()}"
)

if num_tracks is not None:
    log_average_occupancy(solution_graph, num_tracks)
    log_solution_tracks(solution_graph)

if t_min is not None and t_max is not None:
    log_edge_margin_ranking(solver, solution_graph, candidate_graph)

if t_min is not None and t_max is not None:
    retained_ids = sorted(solution_graph.nodes())
    dropped_ids = sorted(set(candidate_graph.nodes()) - set(solution_graph.nodes()))
    expected_dropped = sorted(
        nid for nid in dropped_ids if nid in set(pinned_nodes_false)
    )
    unexpected_dropped = sorted(
        nid for nid in dropped_ids if nid not in set(pinned_nodes_false)
    )
    logger.info("Retained tracklet IDs: %s", retained_ids)
    logger.info("Dropped candidate IDs (%d): %s", len(dropped_ids), dropped_ids)
    logger.info(
        "Unexpected dropped IDs (%d): %s", len(unexpected_dropped), unexpected_dropped
    )
    for src, dst in sorted(solution_graph.edges()):
        logger.info("  Link: %s -> %s", src, dst)


out_path = Path(__file__).parent / "solution" / "solution_centroid.csv"
save_tracklet_csv_data(solution_graph, csv_path, out_path, sequence="cohort_1vs2")
