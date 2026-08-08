"""Post-hoc minimum community audit for ``train_query.py`` results.

This script never trains CMFC.  It loads the original graph, runs one
fixed-seed Louvain partition, joins the community IDs to the saved per-node
query counts, and reports:

1. community exposure ratios;
2. community-distribution total variation distance;
3. Spearman correlation between internal density and exposure.

The Monte Carlo null reproduces the actual sampling design: in each training
event it draws ``queries_per_epoch`` nodes without replacement from the full
candidate pool, then accumulates counts over all sampling events.

Examples
--------
Analyze one result directory:
    python analyze_query_community.py --dataset-name CiteSeer --n 2 --k 1

Analyze every existing query-audit result:
    python analyze_query_community.py --all-existing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import os.path as osp
import pickle
import re
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np

COMMUNITY_OUTPUT_COLUMNS = {
    "community_assignments.csv": [
        "dataset", "node_id", "community_id", "node_degree",
    ],
    "community_structure.csv": [
        "dataset", "community_id", "node_count", "node_share",
        "internal_edges", "internal_density",
    ],
    "community_exposure_runs.csv": [
        "dataset", "setting", "run", "seed", "community_id",
        "node_count", "node_share", "internal_edges", "internal_density",
        "query_count", "query_share", "exposure_ratio",
        "null_exposure_mean", "null_exposure_ci95_low",
        "null_exposure_ci95_high", "outside_null_ci95",
    ],
    "community_metrics_runs.csv": [
        "dataset", "setting", "run", "seed",
        "community_tv", "density_exposure_spearman",
        "null_tv_mean", "null_tv_ci95_low", "null_tv_ci95_high",
        "tv_outside_null_ci95",
        "null_spearman_mean", "null_spearman_ci95_low",
        "null_spearman_ci95_high", "spearman_outside_null_ci95",
    ],
    "community_exposure_summary_95ci.csv": [
        "dataset", "setting", "community_id", "node_count",
        "internal_density", "n_runs", "mean_exposure_ratio",
        "ci95_half_width", "ci95_low", "ci95_high",
        "null_exposure_mean", "null_exposure_ci95_low",
        "null_exposure_ci95_high",
    ],
    "community_metrics_summary_95ci.csv": [
        "dataset", "setting", "metric", "n_runs", "mean",
        "ci95_half_width", "ci95_low", "ci95_high",
        "null_mean", "null_ci95_low", "null_ci95_high",
    ],
}


@dataclass(frozen=True)
class CommunityOptions:
    output_root: str
    dataset_root: str
    all_existing: bool
    community_seed: int
    resolution: float
    louvain_threshold: float
    null_simulations: int
    null_seed: int
    null_event_chunk: int


@dataclass(frozen=True)
class CommunityPartition:
    node_community: np.ndarray
    node_degrees: np.ndarray
    community_sizes: np.ndarray
    internal_edges: np.ndarray
    densities: np.ndarray
    edge_count: int


@dataclass(frozen=True)
class NullDistribution:
    exposure_values: np.ndarray
    exposure_mean: np.ndarray
    exposure_low: np.ndarray
    exposure_high: np.ndarray
    tv_values: np.ndarray
    spearman_values: np.ndarray


@dataclass(frozen=True)
class ExperimentSpec:
    dataset_name: str
    n: int
    k: int


def parse_options(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc Louvain community audit for train_query.py results. "
            "This script does not import PyTorch or retrain CMFC."
        )
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--dataset-root",
        default=osp.join(
            osp.dirname(osp.realpath(__file__)),
            "dataset",
        ),
    )
    parser.add_argument(
        "--query-output-dir",
        default=osp.join("results", "query_audit"),
    )
    parser.add_argument("--all-existing", action="store_true")
    parser.add_argument("--community-seed", type=int, default=42)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument(
        "--louvain-threshold",
        type=float,
        default=1e-7,
    )
    parser.add_argument("--null-simulations", type=int, default=1000)
    parser.add_argument("--null-seed", type=int, default=2025)
    parser.add_argument("--null-event-chunk", type=int, default=25)
    parsed = parser.parse_args(argv)

    if parsed.null_simulations <= 0:
        raise ValueError("--null-simulations must be positive.")
    if parsed.null_event_chunk <= 0:
        raise ValueError("--null-event-chunk must be positive.")
    if parsed.community_resolution <= 0:
        raise ValueError("--community-resolution must be positive.")
    if parsed.louvain_threshold <= 0:
        raise ValueError("--louvain-threshold must be positive.")
    if not parsed.all_existing and (
        parsed.dataset_name is None
        or parsed.n is None
        or parsed.k is None
    ):
        parser.error(
            "single-experiment mode requires --dataset-name, --n, and --k"
        )

    options = CommunityOptions(
        output_root=parsed.query_output_dir,
        dataset_root=osp.abspath(parsed.dataset_root),
        all_existing=parsed.all_existing,
        community_seed=parsed.community_seed,
        resolution=parsed.community_resolution,
        louvain_threshold=parsed.louvain_threshold,
        null_simulations=parsed.null_simulations,
        null_seed=parsed.null_seed,
        null_event_chunk=parsed.null_event_chunk,
    )
    experiment = (
        None
        if parsed.all_existing
        else ExperimentSpec(parsed.dataset_name, parsed.n, parsed.k)
    )
    return options, experiment


def task_setting(args):
    return f"{args.n}-way {args.k}-shot"


def experiment_dir(args, output_root):
    setting_slug = f"{args.n}-way_{args.k}-shot"
    return osp.abspath(
        osp.join(output_root, args.dataset_name, setting_slug)
    )


def finite_or_blank(value):
    value = float(value)
    return value if math.isfinite(value) else ""


def mean_ci95(values):
    values = np.asarray(
        [float(value) for value in values if value != ""],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(values.mean())
    half_width = (
        0.0
        if values.size == 1
        else float(1.96 * values.std(ddof=1) / math.sqrt(values.size))
    )
    return (
        mean,
        half_width,
        mean - half_width,
        mean + half_width,
        int(values.size),
    )


def rankdata(values):
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values)
            and sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def spearman_correlation(first, second):
    first_rank = rankdata(first)
    second_rank = rankdata(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def initialize_csv(path, columns):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def initialize_outputs(output_dir):
    paths = {}
    for filename, columns in COMMUNITY_OUTPUT_COLUMNS.items():
        path = osp.join(output_dir, filename)
        initialize_csv(path, columns)
        paths[filename] = path
    return paths


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_sampling_design(output_dir):
    path = osp.join(output_dir, "sampling_runs.csv")
    if not osp.exists(path):
        raise FileNotFoundError(
            f"Missing sampling audit file: {path}. "
            "Run train_query.py before the community audit."
        )
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No sampling runs found in {path}.")

    designs = {
        (
            int(row["num_nodes"]),
            int(row["queries_per_epoch"]),
            int(row["sampling_events"]),
        )
        for row in rows
    }
    if len(designs) != 1:
        raise ValueError(
            "All runs in one result directory must use the same node count, "
            "queries per epoch, and number of sampling events."
        )
    num_nodes, queries_per_event, sampling_events = designs.pop()
    if queries_per_event > num_nodes:
        raise ValueError(
            "queries_per_epoch cannot exceed the candidate pool size."
        )
    expected_total = queries_per_event * sampling_events
    inconsistent_runs = [
        row["run"] for row in rows
        if int(row["total_query_selections"]) != expected_total
    ]
    if inconsistent_runs:
        raise ValueError(
            "The saved total query count does not equal "
            "queries_per_epoch * sampling_events for runs: "
            + ", ".join(inconsistent_runs)
        )
    run_designs = {
        int(row["run"]): {
            "seed": int(row["seed"]),
            "total_query_selections": int(
                row["total_query_selections"]
            ),
        }
        for row in rows
    }
    return (
        num_nodes,
        queries_per_event,
        sampling_events,
        run_designs,
    )


def read_node_query_counts(output_dir, num_nodes, run_designs):
    path = osp.join(output_dir, "node_exposure_runs.csv")
    if not osp.exists(path):
        raise FileNotFoundError(
            f"Missing node exposure file: {path}. "
            "The community audit needs per-node query counts."
        )

    counts_by_run = {
        run: np.zeros(num_nodes, dtype=np.int64)
        for run in run_designs
    }
    seen_by_run = {
        run: np.zeros(num_nodes, dtype=bool)
        for run in run_designs
    }
    saved_degrees = np.full(num_nodes, -1, dtype=np.int64)

    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run = int(row["run"])
            if run not in counts_by_run:
                raise ValueError(
                    f"Node exposure file contains unknown run {run}."
                )
            node_id = int(row["node_id"])
            if node_id < 0 or node_id >= num_nodes:
                raise ValueError(f"Invalid node_id={node_id}.")
            if seen_by_run[run][node_id]:
                raise ValueError(
                    f"Duplicate node_id={node_id} in run {run}."
                )
            seen_by_run[run][node_id] = True
            counts_by_run[run][node_id] = int(row["query_count"])
            degree = int(row["node_degree"])
            if saved_degrees[node_id] not in (-1, degree):
                raise ValueError(
                    f"Inconsistent saved degree for node {node_id}."
                )
            saved_degrees[node_id] = degree

    for run, counts in counts_by_run.items():
        if not seen_by_run[run].all():
            missing = int((~seen_by_run[run]).sum())
            raise ValueError(
                f"Run {run} is missing {missing} node exposure rows."
            )
        expected_total = run_designs[run]["total_query_selections"]
        if int(counts.sum()) != expected_total:
            raise ValueError(
                f"Run {run} query counts sum to {counts.sum()}, "
                f"expected {expected_total}."
            )
    return counts_by_run, saved_degrees


def canonical_undirected_edges(
    row,
    col,
    num_nodes,
):
    """Coalesce directed/raw edges into unique undirected non-self edges."""
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    valid = (
        (row >= 0)
        & (row < num_nodes)
        & (col >= 0)
        & (col < num_nodes)
        & (row != col)
    )
    lower = np.minimum(row[valid], col[valid])
    upper = np.maximum(row[valid], col[valid])
    packed = np.unique(lower * np.int64(num_nodes) + upper)
    return (
        (packed // num_nodes).astype(np.int64),
        (packed % num_nodes).astype(np.int64),
    )


def load_planetoid_graph(path, num_nodes):
    with open(path, "rb") as handle:
        graph_dict = pickle.load(handle, encoding="latin1")
    sources = []
    targets = []
    for source, neighbors in graph_dict.items():
        sources.extend([int(source)] * len(neighbors))
        targets.extend(int(target) for target in neighbors)
    return canonical_undirected_edges(
        sources,
        targets,
        num_nodes,
    )


def load_npz_graph(path, expected_num_nodes):
    with np.load(path, allow_pickle=True) as archive:
        keys = set(archive.files)
        if {"adj_indices", "adj_indptr", "adj_shape"} <= keys:
            indices = np.asarray(
                archive["adj_indices"],
                dtype=np.int64,
            )
            indptr = np.asarray(
                archive["adj_indptr"],
                dtype=np.int64,
            )
            shape = tuple(
                int(value) for value in archive["adj_shape"]
            )
            if shape != (expected_num_nodes, expected_num_nodes):
                raise ValueError(
                    f"Adjacency shape in {path} is {shape}, expected "
                    f"({expected_num_nodes}, {expected_num_nodes})."
                )
            row = np.repeat(
                np.arange(expected_num_nodes, dtype=np.int64),
                np.diff(indptr),
            )
            col = indices
        elif "edge_index" in keys:
            edge_index = np.asarray(
                archive["edge_index"],
                dtype=np.int64,
            )
            if edge_index.ndim != 2:
                raise ValueError(
                    f"edge_index in {path} must be two-dimensional."
                )
            if edge_index.shape[0] == 2:
                row, col = edge_index
            elif edge_index.shape[1] == 2:
                row, col = edge_index.T
            else:
                raise ValueError(
                    f"Unrecognized edge_index shape {edge_index.shape}."
                )
        else:
            raise ValueError(
                f"Cannot find adjacency arrays in {path}. "
                f"Available keys: {sorted(keys)}"
            )
    return canonical_undirected_edges(
        row,
        col,
        expected_num_nodes,
    )


def load_original_edges(
    dataset_name,
    dataset_root,
    expected_num_nodes,
):
    planetoid_names = {
        "Cora": "cora",
        "CiteSeer": "citeseer",
        "PubMed": "pubmed",
    }
    if dataset_name in planetoid_names:
        raw_name = planetoid_names[dataset_name]
        path = osp.join(
            dataset_root,
            dataset_name,
            "raw",
            f"ind.{raw_name}.graph",
        )
        loader = load_planetoid_graph
    else:
        npz_paths = {
            "Computers": osp.join(
                dataset_root,
                "Computers",
                "raw",
                "amazon_electronics_computers.npz",
            ),
            "Photo": osp.join(
                dataset_root,
                "Photo",
                "raw",
                "amazon_electronics_photo.npz",
            ),
            "CS": osp.join(
                dataset_root,
                "CS",
                "raw",
                "ms_academic_cs.npz",
            ),
            "CoraFull": osp.join(
                dataset_root,
                "CoraFull",
                "cora",
                "raw",
                "cora.npz",
            ),
        }
        if dataset_name not in npz_paths:
            raise ValueError(
                f"Unsupported raw graph format for {dataset_name!r}."
            )
        path = npz_paths[dataset_name]
        loader = load_npz_graph

    if not osp.exists(path):
        raise FileNotFoundError(
            f"Raw graph file not found: {path}"
        )
    return loader(path, expected_num_nodes)


def detect_communities(args, options, expected_num_nodes):
    """Load the original graph and run deterministic NetworkX Louvain."""
    if not hasattr(nx.community, "louvain_communities"):
        raise RuntimeError(
            "This analysis requires NetworkX with "
            "community.louvain_communities (NetworkX >= 2.8)."
        )
    edge_source, edge_target = load_original_edges(
        args.dataset_name,
        options.dataset_root,
        expected_num_nodes,
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(expected_num_nodes))
    graph.add_edges_from(zip(edge_source.tolist(), edge_target.tolist()))

    communities = nx.community.louvain_communities(
        graph,
        weight=None,
        resolution=options.resolution,
        threshold=options.louvain_threshold,
        seed=options.community_seed,
    )
    communities = sorted(
        (sorted(int(node) for node in community) for community in communities),
        key=lambda community: community[0],
    )
    node_community = np.full(expected_num_nodes, -1, dtype=np.int64)
    for community_id, nodes in enumerate(communities):
        node_community[nodes] = community_id
    if np.any(node_community < 0):
        raise RuntimeError("Louvain did not assign every graph node.")

    community_count = len(communities)
    community_sizes = np.bincount(
        node_community,
        minlength=community_count,
    ).astype(np.int64)
    internal_mask = (
        node_community[edge_source] == node_community[edge_target]
    )
    internal_edges = np.bincount(
        node_community[edge_source[internal_mask]],
        minlength=community_count,
    ).astype(np.int64)
    densities = np.full(community_count, np.nan, dtype=float)
    non_singleton = community_sizes > 1
    densities[non_singleton] = (
        2.0 * internal_edges[non_singleton]
        / (
            community_sizes[non_singleton]
            * (community_sizes[non_singleton] - 1)
        )
    )
    node_degrees = np.bincount(
        np.concatenate([edge_source, edge_target]),
        minlength=expected_num_nodes,
    ).astype(np.int64)
    return CommunityPartition(
        node_community=node_community,
        node_degrees=node_degrees,
        community_sizes=community_sizes,
        internal_edges=internal_edges,
        densities=densities,
        edge_count=len(edge_source),
    )


def multivariate_hypergeometric(
    rng,
    category_sizes,
    draws,
    size,
):
    """Use NumPy's implementation, with a vectorized compatibility fallback."""
    if hasattr(rng, "multivariate_hypergeometric"):
        return rng.multivariate_hypergeometric(
            category_sizes,
            draws,
            size=size,
        )

    shape = tuple(size) if isinstance(size, tuple) else (size,)
    result = np.zeros((*shape, len(category_sizes)), dtype=np.int64)
    remaining_draws = np.full(shape, draws, dtype=np.int64)
    remaining_population = int(np.sum(category_sizes))
    for category_id, category_size in enumerate(category_sizes[:-1]):
        selected = rng.hypergeometric(
            int(category_size),
            remaining_population - int(category_size),
            remaining_draws,
        )
        result[..., category_id] = selected
        remaining_draws -= selected
        remaining_population -= int(category_size)
    result[..., -1] = remaining_draws
    return result


def density_exposure_spearman(densities, exposure_ratios):
    valid = np.isfinite(densities) & np.isfinite(exposure_ratios)
    if int(valid.sum()) < 2:
        return float("nan")
    return spearman_correlation(
        densities[valid],
        exposure_ratios[valid],
    )


def sampling_design_null_seed(
    base_seed,
    dataset_name,
    num_nodes,
    queries_per_event,
    sampling_events,
):
    identity = (
        f"{dataset_name}|{num_nodes}|"
        f"{queries_per_event}|{sampling_events}"
    ).encode("utf-8")
    offset = int.from_bytes(
        hashlib.sha256(identity).digest()[:4],
        byteorder="little",
    )
    return (int(base_seed) + offset) % (2**32)


def simulate_uniform_null(
    partition,
    queries_per_event,
    sampling_events,
    simulations,
    seed,
    event_chunk,
):
    """Simulate sums of event-level multivariate hypergeometric draws."""
    rng = np.random.default_rng(seed)
    community_count = len(partition.community_sizes)
    accumulated = np.zeros(
        (simulations, community_count),
        dtype=np.int64,
    )
    completed_events = 0
    while completed_events < sampling_events:
        chunk = min(
            event_chunk,
            sampling_events - completed_events,
        )
        event_counts = multivariate_hypergeometric(
            rng,
            partition.community_sizes,
            queries_per_event,
            size=(simulations, chunk),
        )
        accumulated += event_counts.sum(axis=1, dtype=np.int64)
        completed_events += chunk

    total_queries = queries_per_event * sampling_events
    node_shares = (
        partition.community_sizes
        / partition.community_sizes.sum()
    )
    query_shares = accumulated / total_queries
    exposure = query_shares / node_shares[None, :]
    tv_values = 0.5 * np.abs(
        query_shares - node_shares[None, :]
    ).sum(axis=1)
    spearman_values = np.asarray([
        density_exposure_spearman(partition.densities, row)
        for row in exposure
    ])
    return NullDistribution(
        exposure_values=exposure,
        exposure_mean=exposure.mean(axis=0),
        exposure_low=np.quantile(exposure, 0.025, axis=0),
        exposure_high=np.quantile(exposure, 0.975, axis=0),
        tv_values=tv_values,
        spearman_values=spearman_values,
    )


def interval_statistics(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(finite.mean()),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def null_mean_statistics(
    values,
    run_count,
    seed,
    resamples=10000,
    batch_size=250,
):
    """Null interval for a mean over the same number of independent runs."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    valid_columns = np.all(np.isfinite(values), axis=0)
    output_mean = np.full(values.shape[1], np.nan)
    output_low = np.full(values.shape[1], np.nan)
    output_high = np.full(values.shape[1], np.nan)
    if not np.any(valid_columns):
        return output_mean, output_low, output_high

    finite_values = values[:, valid_columns]
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(
        (resamples, finite_values.shape[1]),
        dtype=float,
    )
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(
            0,
            finite_values.shape[0],
            size=(stop - start, run_count),
        )
        resampled_means[start:stop] = finite_values[indices].mean(axis=1)

    output_mean[valid_columns] = finite_values.mean(axis=0)
    output_low[valid_columns] = np.quantile(
        resampled_means,
        0.025,
        axis=0,
    )
    output_high[valid_columns] = np.quantile(
        resampled_means,
        0.975,
        axis=0,
    )
    return output_mean, output_low, output_high


def outside_interval(value, low, high):
    if not all(math.isfinite(item) for item in (value, low, high)):
        return ""
    return int(value < low or value > high)


def structure_rows(args, partition):
    num_nodes = int(partition.community_sizes.sum())
    rows = []
    for community_id, size in enumerate(partition.community_sizes):
        rows.append({
            "dataset": args.dataset_name,
            "community_id": community_id,
            "node_count": int(size),
            "node_share": int(size) / num_nodes,
            "internal_edges": int(
                partition.internal_edges[community_id]
            ),
            "internal_density": finite_or_blank(
                partition.densities[community_id]
            ),
        })
    return rows


def assignment_rows(args, partition):
    return (
        {
            "dataset": args.dataset_name,
            "node_id": node_id,
            "community_id": int(
                partition.node_community[node_id]
            ),
            "node_degree": int(partition.node_degrees[node_id]),
        }
        for node_id in range(len(partition.node_community))
    )


def observed_run_rows(
    args,
    partition,
    null,
    counts_by_run,
    run_designs,
):
    num_nodes = int(partition.community_sizes.sum())
    node_shares = partition.community_sizes / num_nodes
    null_tv_mean, null_tv_low, null_tv_high = interval_statistics(
        null.tv_values
    )
    null_rho_mean, null_rho_low, null_rho_high = interval_statistics(
        null.spearman_values
    )
    exposure_rows = []
    metric_rows = []

    for run in sorted(counts_by_run):
        node_counts = counts_by_run[run]
        total_queries = int(node_counts.sum())
        community_counts = np.bincount(
            partition.node_community,
            weights=node_counts,
            minlength=len(partition.community_sizes),
        ).astype(np.int64)
        query_shares = community_counts / total_queries
        exposures = query_shares / node_shares
        tv = float(0.5 * np.abs(query_shares - node_shares).sum())
        rho = density_exposure_spearman(
            partition.densities,
            exposures,
        )
        seed = run_designs[run]["seed"]

        for community_id, size in enumerate(partition.community_sizes):
            exposure_rows.append({
                "dataset": args.dataset_name,
                "setting": task_setting(args),
                "run": run,
                "seed": seed,
                "community_id": community_id,
                "node_count": int(size),
                "node_share": node_shares[community_id],
                "internal_edges": int(
                    partition.internal_edges[community_id]
                ),
                "internal_density": finite_or_blank(
                    partition.densities[community_id]
                ),
                "query_count": int(community_counts[community_id]),
                "query_share": query_shares[community_id],
                "exposure_ratio": exposures[community_id],
                "null_exposure_mean": null.exposure_mean[community_id],
                "null_exposure_ci95_low": null.exposure_low[community_id],
                "null_exposure_ci95_high": null.exposure_high[
                    community_id
                ],
                "outside_null_ci95": outside_interval(
                    exposures[community_id],
                    null.exposure_low[community_id],
                    null.exposure_high[community_id],
                ),
            })

        metric_rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "run": run,
            "seed": seed,
            "community_tv": tv,
            "density_exposure_spearman": finite_or_blank(rho),
            "null_tv_mean": null_tv_mean,
            "null_tv_ci95_low": null_tv_low,
            "null_tv_ci95_high": null_tv_high,
            "tv_outside_null_ci95": outside_interval(
                tv,
                null_tv_low,
                null_tv_high,
            ),
            "null_spearman_mean": finite_or_blank(null_rho_mean),
            "null_spearman_ci95_low": finite_or_blank(null_rho_low),
            "null_spearman_ci95_high": finite_or_blank(null_rho_high),
            "spearman_outside_null_ci95": outside_interval(
                rho,
                null_rho_low,
                null_rho_high,
            ),
        })
    return exposure_rows, metric_rows


def build_exposure_summary(
    args,
    partition,
    null,
    exposure_rows,
    null_mean_seed,
):
    run_count = len({row["run"] for row in exposure_rows})
    null_mean, null_low, null_high = null_mean_statistics(
        null.exposure_values,
        run_count,
        null_mean_seed,
    )
    rows = []
    for community_id, size in enumerate(partition.community_sizes):
        values = [
            row["exposure_ratio"]
            for row in exposure_rows
            if row["community_id"] == community_id
        ]
        mean, half, low, high, count = mean_ci95(values)
        rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "community_id": community_id,
            "node_count": int(size),
            "internal_density": finite_or_blank(
                partition.densities[community_id]
            ),
            "n_runs": count,
            "mean_exposure_ratio": finite_or_blank(mean),
            "ci95_half_width": finite_or_blank(half),
            "ci95_low": finite_or_blank(low),
            "ci95_high": finite_or_blank(high),
            "null_exposure_mean": null_mean[community_id],
            "null_exposure_ci95_low": null_low[community_id],
            "null_exposure_ci95_high": null_high[community_id],
        })
    return rows


def build_metric_summary(args, null, metric_rows, null_mean_seed):
    run_count = len(metric_rows)
    null_values = {
        "community_tv": null.tv_values,
        "density_exposure_spearman": null.spearman_values,
    }
    rows = []
    for metric in (
        "community_tv",
        "density_exposure_spearman",
    ):
        mean, half, low, high, count = mean_ci95([
            row[metric] for row in metric_rows
        ])
        metric_null_values = np.asarray(
            null_values[metric],
            dtype=float,
        )
        metric_null_values = metric_null_values[
            np.isfinite(metric_null_values)
        ]
        null_mean_values, null_low_values, null_high_values = (
            null_mean_statistics(
                metric_null_values,
                run_count,
                null_mean_seed + len(rows) + 1,
            )
        )
        null_mean = null_mean_values[0]
        null_low = null_low_values[0]
        null_high = null_high_values[0]
        rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "metric": metric,
            "n_runs": count,
            "mean": finite_or_blank(mean),
            "ci95_half_width": finite_or_blank(half),
            "ci95_low": finite_or_blank(low),
            "ci95_high": finite_or_blank(high),
            "null_mean": finite_or_blank(null_mean),
            "null_ci95_low": finite_or_blank(null_low),
            "null_ci95_high": finite_or_blank(null_high),
        })
    return rows


def write_metadata(
    output_dir,
    args,
    options,
    partition,
    queries_per_event,
    sampling_events,
    null_seed,
):
    singleton_count = int((partition.community_sizes == 1).sum())
    metadata = {
        "analysis": "minimum post-hoc community query audit",
        "dataset": args.dataset_name,
        "setting": task_setting(args),
        "training_performed": False,
        "community_detection": {
            "algorithm": "Louvain",
            "implementation": "networkx.community.louvain_communities",
            "networkx_version": nx.__version__,
            "resolution": options.resolution,
            "threshold": options.louvain_threshold,
            "random_seed": options.community_seed,
            "weight": None,
            "num_communities": len(partition.community_sizes),
            "num_singleton_communities": singleton_count,
            "num_nodes": int(partition.community_sizes.sum()),
            "num_edges": partition.edge_count,
        },
        "density": {
            "definition": "2 * internal_edges / (n * (n - 1))",
            "singleton_handling": (
                "density undefined; singleton communities excluded from "
                "density-exposure Spearman only"
            ),
        },
        "uniform_null": {
            "distribution": (
                "sum of event-level multivariate hypergeometric draws"
            ),
            "candidate_nodes": int(partition.community_sizes.sum()),
            "queries_per_event": queries_per_event,
            "sampling_events": sampling_events,
            "simulations": options.null_simulations,
            "base_seed": options.null_seed,
            "sampling_design_seed": null_seed,
            "interval": "empirical 2.5th and 97.5th percentiles",
        },
        "run_level_summary_interval": (
            "mean +/- 1.96 * sample SD / sqrt(number of runs)"
        ),
        "summary_null_interval": (
            "2.5th and 97.5th percentiles of 10,000 resampled null "
            "means, each using the same number of runs as the observation"
        ),
    }
    with open(
        osp.join(output_dir, "community_analysis.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def analyze_one(
    args,
    options,
    partition_cache=None,
    null_cache=None,
):
    output_dir = experiment_dir(args, options.output_root)
    (
        num_nodes,
        queries_per_event,
        sampling_events,
        run_designs,
    ) = read_sampling_design(output_dir)
    counts_by_run, saved_degrees = read_node_query_counts(
        output_dir,
        num_nodes,
        run_designs,
    )

    print(
        f"\nCommunity audit | {args.dataset_name} | "
        f"{task_setting(args)} | {len(run_designs)} runs"
    )
    if partition_cache is None:
        partition_cache = {}
    if null_cache is None:
        null_cache = {}
    partition_key = (
        args.dataset_name,
        options.community_seed,
        options.resolution,
        options.louvain_threshold,
    )
    if partition_key not in partition_cache:
        partition_cache[partition_key] = detect_communities(
            args,
            options,
            num_nodes,
        )
    partition = partition_cache[partition_key]
    if int(partition.community_sizes.sum()) != num_nodes:
        raise ValueError(
            "Cached community partition has a different node count."
        )
    if not np.array_equal(partition.node_degrees, saved_degrees):
        mismatch = int(
            np.count_nonzero(partition.node_degrees != saved_degrees)
        )
        raise ValueError(
            f"Original graph degree check failed for {mismatch} nodes. "
            "The saved audit and loaded graph are not identical."
        )

    null_seed = sampling_design_null_seed(
        options.null_seed,
        args.dataset_name,
        num_nodes,
        queries_per_event,
        sampling_events,
    )
    null_key = (
        partition_key,
        queries_per_event,
        sampling_events,
        options.null_simulations,
        null_seed,
        options.null_event_chunk,
    )
    if null_key not in null_cache:
        print(
            f"Detected {len(partition.community_sizes)} communities; "
            f"simulating {options.null_simulations} uniform null runs..."
        )
        null_cache[null_key] = simulate_uniform_null(
            partition,
            queries_per_event,
            sampling_events,
            options.null_simulations,
            null_seed,
            options.null_event_chunk,
        )
    else:
        print(
            f"Detected {len(partition.community_sizes)} communities; "
            "reusing the matching uniform null distribution."
        )
    null = null_cache[null_key]
    exposure_rows, metric_rows = observed_run_rows(
        args,
        partition,
        null,
        counts_by_run,
        run_designs,
    )
    exposure_summary = build_exposure_summary(
        args,
        partition,
        null,
        exposure_rows,
        null_seed + 101,
    )
    metric_summary = build_metric_summary(
        args,
        null,
        metric_rows,
        null_seed + 202,
    )
    paths = initialize_outputs(output_dir)
    write_csv(
        paths["community_assignments.csv"],
        COMMUNITY_OUTPUT_COLUMNS["community_assignments.csv"],
        assignment_rows(args, partition),
    )
    write_csv(
        paths["community_structure.csv"],
        COMMUNITY_OUTPUT_COLUMNS["community_structure.csv"],
        structure_rows(args, partition),
    )
    write_csv(
        paths["community_exposure_runs.csv"],
        COMMUNITY_OUTPUT_COLUMNS["community_exposure_runs.csv"],
        exposure_rows,
    )
    write_csv(
        paths["community_metrics_runs.csv"],
        COMMUNITY_OUTPUT_COLUMNS["community_metrics_runs.csv"],
        metric_rows,
    )
    write_csv(
        paths["community_exposure_summary_95ci.csv"],
        COMMUNITY_OUTPUT_COLUMNS[
            "community_exposure_summary_95ci.csv"
        ],
        exposure_summary,
    )
    write_csv(
        paths["community_metrics_summary_95ci.csv"],
        COMMUNITY_OUTPUT_COLUMNS[
            "community_metrics_summary_95ci.csv"
        ],
        metric_summary,
    )
    write_metadata(
        output_dir,
        args,
        options,
        partition,
        queries_per_event,
        sampling_events,
        null_seed,
    )
    print(f"Community audit saved to: {output_dir}")
    return output_dir


def discover_existing_experiments(output_root):
    if not osp.isdir(output_root):
        raise FileNotFoundError(
            f"Query audit root does not exist: {output_root}"
        )
    experiments = []
    for current_dir, _, filenames in os.walk(output_root):
        if (
            "sampling_runs.csv" not in filenames
            or "node_exposure_runs.csv" not in filenames
        ):
            continue
        rows = read_csv_rows(osp.join(current_dir, "sampling_runs.csv"))
        if not rows:
            continue
        dataset_name = rows[0]["dataset"]
        match = re.fullmatch(
            r"(\d+)-way\s+(\d+)-shot",
            rows[0]["setting"],
        )
        if match is None:
            raise ValueError(
                f"Cannot parse task setting in {current_dir}: "
                f"{rows[0]['setting']!r}"
            )
        experiments.append(
            (dataset_name, int(match.group(1)), int(match.group(2)))
        )
    return sorted(set(experiments))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    options, selected_experiment = parse_options(argv)
    if options.all_existing:
        experiments = discover_existing_experiments(
            options.output_root
        )
        if not experiments:
            raise FileNotFoundError(
                "No completed query-audit experiments were found."
        )
        partition_cache = {}
        null_cache = {}
        for dataset_name, n_way, k_shot in experiments:
            args = ExperimentSpec(
                dataset_name=dataset_name,
                n=n_way,
                k=k_shot,
            )
            analyze_one(
                args,
                options,
                partition_cache,
                null_cache,
            )
        return

    analyze_one(selected_experiment, options)


if __name__ == "__main__":
    main()
