import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

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
from data_utils.save_tracklet_csv_data import save_solution_keypoints_csv

logger = logging.getLogger(__name__)

# Bump when create_tracklet_candidate_graph or this script changes the
# semantics of the cached pickle so old caches auto-invalidate.
CACHE_VERSION = 1

_REQUIRED_KEYS = {
    "csv_path",
    "embeddings_path",
    "embeddings_attribute_prefix",
    "max_time_gap",
    "max_spatial_distance",
    "node_attributes",
    "num_tracks",
    "set_exact_track_count",
    "set_exact_active_tracklets_per_frame",
    "solution_csv",
    "use_spatial_distance",
    "use_id_embedding_distance",
}

_OPTIONAL_KEYS = {
    "neighbors_scale",
    "num_neighbors",
    "max_time_gap_past",
    "t_min",
    "t_max",
    "group",
    "candidate_graph_cache",
    "pinned_nodes_true",
    "pinned_nodes_false",
    "pinned_chains",
    "weights_yaml",
    "num_keypoints",
}

# Keys removed in recent script changes — call out by name to make migration easy.
_REMOVED_KEYS = {
    "edge_attributes": "now derived from use_spatial_distance / use_id_embedding_distance / num_keypoints",
    "output_dir": "use `solution_csv` (full path) instead",
    "weights_data": "use `weights_yaml: <path>` instead",
    "tracklet_centroids_csv": "raw centroid dumps were removed",
    "tracklet_centroids_filtered_csv": "raw centroid dumps were removed",
}


def _validate_config(config: dict) -> None:
    """Fail fast with all config problems in one message."""
    keys = set(config.keys())
    problems = []

    missing = _REQUIRED_KEYS - keys
    if missing:
        problems.append(f"missing required keys: {sorted(missing)}")

    for k, reason in _REMOVED_KEYS.items():
        if k in keys:
            problems.append(f"`{k}` was removed — {reason}")

    unknown = keys - (_REQUIRED_KEYS | _OPTIONAL_KEYS | set(_REMOVED_KEYS))
    if unknown:
        problems.append(f"unknown keys (typo?): {sorted(unknown)}")

    if "neighbors_scale" not in keys and "num_neighbors" not in keys:
        problems.append("either `neighbors_scale` or `num_neighbors` must be set")

    if config.get("use_spatial_distance") and "num_keypoints" not in keys:
        problems.append("`use_spatial_distance: true` requires `num_keypoints`")

    if (
        config.get("use_id_embedding_distance")
        and config.get("embeddings_path") is None
    ):
        problems.append(
            "`use_id_embedding_distance: true` requires `embeddings_path` to be set"
        )

    if (
        "use_spatial_distance" in keys
        and "use_id_embedding_distance" in keys
        and not config["use_spatial_distance"]
        and not config["use_id_embedding_distance"]
    ):
        problems.append(
            "at least one of `use_spatial_distance` / `use_id_embedding_distance` must be true"
        )

    if problems:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(problems))


def _derive_edge_attributes(
    use_spatial_distance: bool,
    use_id_embedding_distance: bool,
    num_keypoints: Optional[int],
) -> list:
    """Build the edge-attribute list from the mode flags. `t_gap` is always last."""
    attrs = []
    if use_spatial_distance:
        attrs += [f"kp_{i}_distance" for i in range(num_keypoints)]
    if use_id_embedding_distance:
        attrs.append("id_distance")
    attrs.append("t_gap")
    return attrs


def _expected_weight_keys(edge_attributes: list, node_attributes: list) -> set:
    """The full set of motile weight keys the solver will read."""
    keys = set()
    for attr in node_attributes:
        keys.add(f"Node Selection {attr}_weight")
        keys.add(f"Node Selection {attr}_constant")
    for attr in edge_attributes:
        keys.add(f"Edge Selection {attr}_regular_weight")
        keys.add(f"Edge Selection {attr}_regular_constant")
    keys.add("Appear_weight")
    keys.add("Appear_constant")
    keys.add("Disappear_constant")
    return keys


def _default_weights(edge_attributes: list, node_attributes: list) -> dict:
    """Hand-set defaults used when no `weights_yaml` is provided."""
    w = {}
    for attr in node_attributes:
        w[f"Node Selection {attr}_weight"] = -1
        w[f"Node Selection {attr}_constant"] = 0
    for attr in edge_attributes:
        w[f"Edge Selection {attr}_regular_weight"] = 1
        w[f"Edge Selection {attr}_regular_constant"] = 0
    w["Appear_weight"] = 0
    w["Appear_constant"] = 1
    w["Disappear_constant"] = 1
    return w


def _load_weights(
    weights_yaml: Optional[str],
    edge_attributes: list,
    node_attributes: list,
) -> dict:
    expected = _expected_weight_keys(edge_attributes, node_attributes)

    if weights_yaml is None:
        weights_data = _default_weights(edge_attributes, node_attributes)
        logger.info(
            "Weights: using built-in defaults (%d entries) — no weights_yaml set.",
            len(weights_data),
        )
    else:
        weights_data = load_config(str(Path(weights_yaml).expanduser()))
        logger.info(
            "Weights: loaded from %s (%d entries).", weights_yaml, len(weights_data)
        )

    missing = expected - set(weights_data.keys())
    extra = set(weights_data.keys()) - expected
    if missing or extra:
        raise ValueError(
            "Weights file does not match derived edge/node attributes:\n"
            f"  missing: {sorted(missing)}\n"
            f"  extra:   {sorted(extra)}"
        )
    return weights_data


def _mtime(path: Optional[str]) -> Optional[float]:
    if path is None:
        return None
    try:
        return Path(path).stat().st_mtime
    except FileNotFoundError:
        return None


def _candidate_graph_cache_key(
    csv_path: str,
    embeddings_path: Optional[str],
    embeddings_attribute_prefix: str,
    num_neighbors: int,
    max_spatial_distance,
    max_time_gap: int,
    max_time_gap_past: int,
    t_min,
    t_max,
) -> Tuple:
    return (
        CACHE_VERSION,
        str(csv_path),
        _mtime(csv_path),
        str(embeddings_path) if embeddings_path is not None else None,
        _mtime(embeddings_path),
        embeddings_attribute_prefix,
        int(num_neighbors),
        max_spatial_distance,
        int(max_time_gap),
        int(max_time_gap_past),
        t_min,
        t_max,
    )


def _load_or_build_candidate_graph(
    cache_path: Optional[Path],
    cache_key: Tuple,
    builder_kwargs: dict,
):
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == cache_key:
                logger.info(
                    "Candidate graph: cache hit (%s) — %d nodes, %d edges.",
                    cache_path,
                    cached["graph"].number_of_nodes(),
                    cached["graph"].number_of_edges(),
                )
                return cached["graph"]
            logger.info(
                "Candidate graph: cache key mismatch at %s — rebuilding.",
                cache_path,
            )
        except Exception as e:  # corrupt pickle, version mismatch, etc.
            logger.warning(
                "Candidate graph: cache at %s could not be loaded (%s) — rebuilding.",
                cache_path,
                e,
            )

    logger.info("Building candidate graph…")
    graph = create_tracklet_candidate_graph(**builder_kwargs)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"key": cache_key, "graph": graph},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Candidate graph: cached to %s", cache_path)

    return graph


def _warn_skipped_pins(
    candidate_graph,
    pinned_nodes_true: list,
    pinned_nodes_false: list,
    pinned_chains: list,
) -> None:
    """Log a warning for any pinned IDs/edges that aren't in the candidate graph.

    Silent skips here previously masked wrong solutions when constraints referenced
    tracklet IDs that had been renamed/cut/dropped upstream.
    """
    graph_nodes = set(candidate_graph.nodes)

    missing_true = [nid for nid in pinned_nodes_true if nid not in graph_nodes]
    missing_false = [nid for nid in pinned_nodes_false if nid not in graph_nodes]

    missing_chain_nodes = []
    missing_chain_edges = []
    for chain in pinned_chains:
        for nid in chain:
            if nid not in graph_nodes:
                missing_chain_nodes.append(nid)
        for u, v in zip(chain[:-1], chain[1:]):
            if not candidate_graph.has_edge(u, v):
                missing_chain_edges.append((u, v))

    if missing_true:
        logger.warning(
            "Skipped %d pinned_nodes_true (force-in) not in candidate graph: %s",
            len(missing_true),
            missing_true,
        )
    if missing_false:
        logger.warning(
            "Skipped %d pinned_nodes_false (force-out) not in candidate graph: %s",
            len(missing_false),
            missing_false,
        )
    if missing_chain_nodes:
        logger.warning(
            "Skipped %d pinned_chain nodes not in candidate graph: %s",
            len(missing_chain_nodes),
            missing_chain_nodes,
        )
    if missing_chain_edges:
        logger.warning(
            "Skipped %d pinned_chain edges not in candidate graph: %s",
            len(missing_chain_edges),
            missing_chain_edges,
        )


def _log_infeasibility_diagnostics(
    err,
    *,
    candidate_graph,
    num_tracks,
    set_exact_track_count,
    pinned_nodes_true,
    pinned_nodes_false,
    pinned_chains,
) -> None:
    n_chain_edges = sum(max(0, len(c) - 1) for c in pinned_chains)
    logger.error("ILP solve failed: %s", err)
    logger.error("Diagnostics:")
    logger.error("  num_tracks: %s", num_tracks)
    logger.error("  set_exact_track_count: %s", set_exact_track_count)
    logger.error("  pinned_nodes_true   (force-in):  %d", len(pinned_nodes_true))
    logger.error("  pinned_nodes_false  (force-out): %d", len(pinned_nodes_false))
    logger.error(
        "  pinned_chains: %d (with %d edges total)", len(pinned_chains), n_chain_edges
    )
    logger.error(
        "  candidate graph: %d nodes, %d edges",
        candidate_graph.number_of_nodes(),
        candidate_graph.number_of_edges(),
    )


def main(config_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(str(config_path))
    print_config(config)
    _validate_config(config)

    csv_path = config["csv_path"]
    embeddings_path = config["embeddings_path"]
    max_time_gap = config["max_time_gap"]
    neighbors_scale = config.get("neighbors_scale", None)
    if neighbors_scale is not None:
        num_neighbors = max(1, round(neighbors_scale * max_time_gap))
        logger.info(
            "num_neighbors computed from neighbors_scale=%.4f * max_time_gap=%d → %d",
            neighbors_scale,
            max_time_gap,
            num_neighbors,
        )
    else:
        num_neighbors = config["num_neighbors"]
    max_spatial_distance = config["max_spatial_distance"]
    node_attributes = config["node_attributes"]
    num_tracks = config["num_tracks"]
    embeddings_attribute_prefix = config["embeddings_attribute_prefix"]
    set_exact_track_count = config["set_exact_track_count"]
    set_exact_active_tracklets_per_frame = config[
        "set_exact_active_tracklets_per_frame"
    ]
    t_min = config.get("t_min", None)
    t_max = config.get("t_max", None)
    max_time_gap_past = config.get("max_time_gap_past", 0)
    group = config.get("group")

    use_spatial_distance = config["use_spatial_distance"]
    use_id_embedding_distance = config["use_id_embedding_distance"]
    num_keypoints = config.get("num_keypoints")
    edge_attributes = _derive_edge_attributes(
        use_spatial_distance,
        use_id_embedding_distance,
        num_keypoints,
    )
    logger.info("Edge attributes (derived): %s", edge_attributes)

    # Output path from the YAML — no implicit defaults from script location.
    # solution_csv: required; full path for the solution keypoints CSV.
    solution_csv = Path(config["solution_csv"]).expanduser()
    solution_csv.parent.mkdir(parents=True, exist_ok=True)

    # Optional candidate-graph cache. Skip caching if the key is absent.
    candidate_graph_cache = config.get("candidate_graph_cache")
    cache_path = (
        Path(candidate_graph_cache).expanduser() if candidate_graph_cache else None
    )

    # Optional weights file. If absent, fall back to hand-set defaults.
    weights_yaml = config.get("weights_yaml")

    # Pin specified nodes and edges before building the TrackGraph.
    # pinned_nodes_true:  tracklet IDs to force INTO the solution (pin=True).
    # pinned_nodes_false: tracklet IDs to force OUT of the solution (pin=False).
    # pinned_chains:      ordered tracklet-ID sequences; all nodes and the
    #                     edges connecting consecutive pairs are pinned True.
    pinned_nodes_true = config.get("pinned_nodes_true", [])
    pinned_nodes_false = config.get("pinned_nodes_false", [])
    pinned_chains = config.get("pinned_chains", [])

    # =========================================================================
    # 1. Build (or load) candidate graph
    # =========================================================================

    builder_kwargs = dict(
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
    cache_key = _candidate_graph_cache_key(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        embeddings_attribute_prefix=embeddings_attribute_prefix,
        num_neighbors=num_neighbors,
        max_spatial_distance=max_spatial_distance,
        max_time_gap=max_time_gap,
        max_time_gap_past=max_time_gap_past,
        t_min=t_min,
        t_max=t_max,
    )
    candidate_graph = _load_or_build_candidate_graph(
        cache_path=cache_path,
        cache_key=cache_key,
        builder_kwargs=builder_kwargs,
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

    _warn_skipped_pins(
        candidate_graph, pinned_nodes_true, pinned_nodes_false, pinned_chains
    )

    for nid in pinned_nodes_true:
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

    # =========================================================================
    # 2. Formulate and solve ILP
    # =========================================================================

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

    weights_data = _load_weights(weights_yaml, edge_attributes, node_attributes)
    weights_array = np.array(list(weights_data.values()))
    logger.info(f"Using weights: {weights_array}")
    solver.weights.from_ndarray(weights_array)
    logger.info("Solving ILP optimization...")
    try:
        solution = solver.solve(verbose=False)
    except Exception as e:
        _log_infeasibility_diagnostics(
            e,
            candidate_graph=candidate_graph,
            num_tracks=num_tracks,
            set_exact_track_count=set_exact_track_count,
            pinned_nodes_true=pinned_nodes_true,
            pinned_nodes_false=pinned_nodes_false,
            pinned_chains=pinned_chains,
        )
        raise
    solution_graph = get_solution_graph(solver, solution)
    logger.info(
        f"# solution nodes: {solution_graph.number_of_nodes()}, "
        f"# solution edges: {solution_graph.number_of_edges()}"
    )

    # =========================================================================
    # 3. Analyse and save solution
    # =========================================================================

    if num_tracks is not None:
        log_average_occupancy(solution_graph, num_tracks)
        log_solution_tracks(solution_graph)

    if t_min is not None and t_max is not None:
        log_edge_margin_ranking(solver, solution_graph, candidate_graph)

    if t_min is not None and t_max is not None:
        retained_ids = sorted(solution_graph.nodes())
        dropped_ids = sorted(set(candidate_graph.nodes()) - set(solution_graph.nodes()))
        unexpected_dropped = sorted(
            nid for nid in dropped_ids if nid not in set(pinned_nodes_false)
        )
        logger.info("Retained tracklet IDs: %s", retained_ids)
        logger.info("Dropped candidate IDs (%d): %s", len(dropped_ids), dropped_ids)
        logger.info(
            "Unexpected dropped IDs (%d): %s",
            len(unexpected_dropped),
            unexpected_dropped,
        )
        for src, dst in sorted(solution_graph.edges()):
            logger.info("  Link: %s -> %s", src, dst)

    save_solution_keypoints_csv(
        solution_graph,
        csv_path,
        solution_csv,
        group=group,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run motile tracklet-stitching ILP solver from a YAML config."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config (e.g. ../configs/2026-05-15.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
