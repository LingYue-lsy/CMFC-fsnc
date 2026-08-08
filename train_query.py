"""Minimum query-sampling fairness and sparse-node robustness experiment.

The CMFC model, losses, optimizer, and update sequence are intentionally kept
identical to ``train.py``.  This entry point only records sufficient audit
statistics and, for CiteSeer/Computers, degree-stratified LMA and predictions.

Examples
--------
Run one configured experiment:
    python train_query.py --dataset-name CiteSeer --n 2 --k 1

Run all ten minimum-protocol dataset/shot combinations:
    python train_query.py --minimum-suite

Use a different output root:
    python train_query.py --query-output-dir results/query_audit
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import os.path as osp
import random
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import sklearn
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import train as base
from args_utils import get_args, pprint_args
from models import CMFC_Fast


DEGREE_GROUPS = ("low", "medium", "high")
DETAILED_DATASETS = {"CiteSeer", "Computers"}
MINIMUM_SUITE = (
    ("Cora", 2, 1),
    ("Cora", 2, 5),
    ("CiteSeer", 2, 1),
    ("CiteSeer", 2, 5),
    ("CoraFull", 5, 1),
    ("CoraFull", 5, 5),
    ("Computers", 2, 1),
    ("Computers", 2, 5),
    ("CS", 5, 1),
    ("CS", 5, 5),
)


@dataclass(frozen=True)
class QueryOptions:
    output_root: str
    minimum_suite: bool


OUTPUT_COLUMNS = {
    "run_metadata.csv": [
        "dataset", "setting", "run", "seed", "train_split",
        "validation_split", "test_split",
    ],
    "sampling_runs.csv": [
        "dataset", "setting", "run", "seed", "num_nodes",
        "queries_per_epoch", "sampling_events", "total_query_selections",
        "spearman_degree_query_count", "class_tv", "final_coverage",
        "theoretical_final_coverage",
    ],
    "degree_exposure_runs.csv": [
        "dataset", "setting", "run", "seed", "degree_group",
        "degree_min", "degree_max", "node_count", "node_share",
        "query_count", "query_share", "exposure_ratio",
    ],
    "class_exposure_runs.csv": [
        "dataset", "setting", "run", "seed", "class_id",
        "node_count", "node_share", "query_count", "query_share",
        "exposure_ratio",
    ],
    "coverage_curve.csv": [
        "dataset", "setting", "run", "seed", "sampling_event",
        "actual_coverage", "theoretical_coverage",
    ],
    "coverage_summary_95ci.csv": [
        "dataset", "setting", "sampling_event", "n_runs",
        "mean_actual_coverage", "ci95_half_width", "ci95_low", "ci95_high",
        "theoretical_coverage",
    ],
    "node_exposure_runs.csv": [
        "dataset", "setting", "run", "seed", "node_id", "node_degree",
        "degree_group", "ground_truth_class", "query_count",
    ],
    "lma_curve.csv": [
        "dataset", "setting", "run", "seed", "epoch", "degree_group",
        "query_count", "matching_supports", "support_links", "lma",
    ],
    "lma_summary_95ci.csv": [
        "dataset", "setting", "epoch", "degree_group", "n_runs",
        "mean_lma", "ci95_half_width", "ci95_low", "ci95_high",
    ],
    "performance_runs.csv": [
        "dataset", "setting", "run", "seed", "overall_accuracy_percent",
        "acc_low_percent", "acc_medium_percent", "acc_high_percent",
        "delta_low_high_percentage_points",
        "lma_final20_low", "lma_final20_medium", "lma_final20_high",
        "delta_lma_low_high",
    ],
    "test_predictions.csv": [
        "dataset", "setting", "run", "seed", "repeat",
        "query_node_id", "node_degree", "degree_group",
        "ground_truth_class", "predicted_class", "correct",
    ],
    "summary_95ci.csv": [
        "dataset", "setting", "metric", "subgroup", "n_runs",
        "mean", "ci95_half_width", "ci95_low", "ci95_high",
    ],
}


def parse_query_options(argv):
    """Remove this script's options before delegating to the base parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--query-output-dir",
        default=osp.join("results", "query_audit"),
    )
    parser.add_argument("--minimum-suite", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    return (
        QueryOptions(
            output_root=options.query_output_dir,
            minimum_suite=options.minimum_suite,
        ),
        remaining,
    )


def load_args(remaining_argv, overrides=()):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining_argv, *overrides]
        return get_args(
            model_name="CMFC",
            dataset_class="Planetoid",
            dataset_name=None,
            custom_key="classification",
        )
    finally:
        sys.argv = old_argv


def task_setting(args):
    return f"{args.n}-way {args.k}-shot"


def experiment_dir(args, output_root):
    setting_slug = f"{args.n}-way_{args.k}-shot"
    return osp.abspath(osp.join(output_root, args.dataset_name, setting_slug))


def initialize_csv(path, columns):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()


def append_csv(path, columns, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writerows(rows)


def initialize_outputs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for filename, columns in OUTPUT_COLUMNS.items():
        path = osp.join(output_dir, filename)
        initialize_csv(path, columns)
        paths[filename] = path
    return paths


def raw_undirected_degrees(adjacency):
    """Return binary degrees, excluding the self-loops added by CMFC."""
    row, col = adjacency.coalesce().indices().cpu()
    non_loop = row != col
    return torch.bincount(
        row[non_loop],
        minlength=adjacency.size(0),
    ).numpy().astype(np.int64)


def degree_group_labels(degrees):
    """Use global 33/67 percentiles while keeping degree ties together."""
    degrees = np.asarray(degrees)
    q33, q67 = np.quantile(degrees, [0.33, 0.67])
    groups = np.full(degrees.shape, "medium", dtype=object)
    groups[degrees <= q33] = "low"
    groups[degrees > q67] = "high"
    return groups


def class_conditional_degree_groups(node_ids, labels, degrees):
    """Assign test nodes to degree groups independently within each class."""
    groups = np.full(len(degrees), "", dtype=object)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    for class_id in np.unique(labels[node_ids]):
        class_nodes = node_ids[labels[node_ids] == class_id]
        groups[class_nodes] = degree_group_labels(degrees[class_nodes])
    return groups


def spearman_correlation(first, second):
    first_rank = rankdata(np.asarray(first))
    second_rank = rankdata(np.asarray(second))
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


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


def group_exposure_rows(
    args,
    run,
    seed,
    degrees,
    groups,
    query_counts,
):
    total_queries = int(query_counts.sum())
    rows = []
    for group in DEGREE_GROUPS:
        mask = groups == group
        node_count = int(mask.sum())
        query_count = int(query_counts[mask].sum())
        node_share = node_count / len(query_counts)
        query_share = query_count / total_queries
        rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "run": run,
            "seed": seed,
            "degree_group": group,
            "degree_min": int(degrees[mask].min()) if node_count else "",
            "degree_max": int(degrees[mask].max()) if node_count else "",
            "node_count": node_count,
            "node_share": node_share,
            "query_count": query_count,
            "query_share": query_share,
            "exposure_ratio": (
                query_share / node_share if node_share > 0 else ""
            ),
        })
    return rows


def class_exposure_rows(args, run, seed, labels, query_counts):
    total_queries = int(query_counts.sum())
    rows = []
    for class_id in np.unique(labels):
        mask = labels == class_id
        node_count = int(mask.sum())
        query_count = int(query_counts[mask].sum())
        node_share = node_count / len(labels)
        query_share = query_count / total_queries
        rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "run": run,
            "seed": seed,
            "class_id": int(class_id),
            "node_count": node_count,
            "node_share": node_share,
            "query_count": query_count,
            "query_share": query_share,
            "exposure_ratio": query_share / node_share,
        })
    return rows


def aggregate_final_lma(lma_rows, epochs):
    final_epoch_count = max(1, int(math.ceil(0.20 * epochs)))
    cutoff = epochs - final_epoch_count
    result = {}
    for group in DEGREE_GROUPS:
        selected = [
            row for row in lma_rows
            if row["degree_group"] == group and row["epoch"] >= cutoff
        ]
        matches = sum(row["matching_supports"] for row in selected)
        links = sum(row["support_links"] for row in selected)
        result[group] = matches / links if links else float("nan")
    return result


def audited_training_run(
    args,
    running_device,
    prepared_graph,
    masks,
    run_number,
    degrees,
    global_groups,
    detailed,
):
    """Train CMFC with the exact base update logic and collect audit data."""
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.m <= 0:
        raise ValueError(f"m must be a positive integer, got {args.m}.")

    model = CMFC_Fast(
        prepared_graph.num_features,
        cfg=args.cfg,
        dropout=args.dropout,
    ).to(running_device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_neg = args.NN
    mask_margin = args.margin1 + args.margin2
    num_task_node = min(
        args.n * args.task_num,
        prepared_graph.num_nodes,
    )
    labels_cpu = prepared_graph.labels_cpu.numpy()
    query_counts = np.zeros(prepared_graph.num_nodes, dtype=np.int64)
    coverage_rows = []
    lma_rows = []

    for epoch in tqdm(
        base.training_epoch_range(args),
        desc=f"run {run_number}",
    ):
        model.train()
        optimiser.zero_grad()
        h_a, h_p, h_a2 = model(
            prepared_graph.features,
            prepared_graph.adjacency,
        )

        task_node_index_np = np.random.choice(
            prepared_graph.num_nodes,
            size=num_task_node,
            replace=False,
        )
        task_node_index = torch.as_tensor(
            task_node_index_np,
            dtype=torch.long,
            device=running_device,
        )
        np.add.at(query_counts, task_node_index_np, 1)

        negative_indices = torch.as_tensor(
            np.stack([
                np.random.permutation(num_task_node)
                for _ in range(num_neg)
            ]),
            dtype=torch.long,
            device=running_device,
        )
        h_a_task = h_a[task_node_index]
        topk_indices = base.select_support_indices(
            h_a_task,
            h_a2,
            args.k,
            args.support_selection,
        )
        neighbor_embeddings = base.aggregate_sampled_neighbors(
            h_a2,
            topk_indices,
            prepared_graph.neighbor_table,
            epoch,
            args.m,
        )
        support_distances, neighbor_distances, negative_distances = (
            base.compute_batched_distances(
                h_a_task,
                h_p,
                neighbor_embeddings,
                topk_indices,
                negative_indices,
            )
        )
        task_loss_mar, task_loss_mar_1, task_mask_margin_n = (
            base.compute_batched_margin_terms(
                support_distances,
                neighbor_distances,
                negative_distances,
                args.margin1,
                mask_margin,
            )
        )
        loss_mar = task_loss_mar / (num_task_node * args.k) * 100
        loss_mar_1 = task_loss_mar_1 / (num_task_node * args.k) * 100
        mask_margin_n = (
            task_mask_margin_n
            / (num_task_node * args.k * num_neg)
            * 100
        )
        loss = (
            loss_mar * args.w_loss1
            + loss_mar_1 * args.w_loss2
            + mask_margin_n * args.w_loss3
        )
        loss.backward()
        optimiser.step()

        if args.save_model:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                osp.join(
                    args.checkpoint_dir,
                    f"{args.dataset_name}_weights.pth",
                ),
            )

        sampling_event = len(coverage_rows) + 1
        actual_coverage = float(np.count_nonzero(query_counts)) / len(
            query_counts
        )
        theoretical_coverage = 1.0 - (
            1.0 - num_task_node / len(query_counts)
        ) ** sampling_event
        coverage_rows.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "run": run_number,
            "seed": seed,
            "sampling_event": sampling_event,
            "actual_coverage": actual_coverage,
            "theoretical_coverage": theoretical_coverage,
        })

        if detailed:
            support_ids = topk_indices.detach().cpu().numpy()
            matches = (
                labels_cpu[support_ids]
                == labels_cpu[task_node_index_np, None]
            )
            selected_groups = global_groups[task_node_index_np]
            for group in DEGREE_GROUPS:
                mask = selected_groups == group
                query_count = int(mask.sum())
                matching_supports = int(matches[mask].sum())
                support_links = query_count * args.k
                lma_rows.append({
                    "dataset": args.dataset_name,
                    "setting": task_setting(args),
                    "run": run_number,
                    "seed": seed,
                    "epoch": epoch,
                    "degree_group": group,
                    "query_count": query_count,
                    "matching_supports": matching_supports,
                    "support_links": support_links,
                    "lma": (
                        matching_supports / support_links
                        if support_links else ""
                    ),
                })

    lma_final = (
        aggregate_final_lma(lma_rows, int(args.epochs))
        if detailed else {}
    )
    return model, query_counts, coverage_rows, lma_rows, lma_final


def evaluate_with_degree_predictions(
    model,
    prepared_graph,
    test_mask,
    args,
    run_number,
    seed,
    degrees,
):
    """Mirror base evaluation and additionally retain each query prediction."""
    model.eval()
    with torch.no_grad():
        _, propagated_embeddings, _ = model.embed(
            prepared_graph.features,
            prepared_graph.adjacency,
        )
        normalized = propagated_embeddings / (
            propagated_embeddings.norm(dim=1)[:, None]
        )
        embeddings = normalized.cpu().numpy()

    eval_indices = torch.nonzero(
        test_mask,
        as_tuple=False,
    ).view(-1).cpu().numpy()
    labels = prepared_graph.labels_cpu.numpy()
    eval_labels = labels[eval_indices]
    available_classes = np.unique(eval_labels)
    n_way = min(args.n, len(available_classes))
    class_indices = {
        class_id: eval_indices[eval_labels == class_id]
        for class_id in available_classes
    }
    degree_groups = class_conditional_degree_groups(
        eval_indices,
        labels,
        degrees,
    )

    prediction_rows = []
    repeat_accuracies = []
    group_correct = {group: 0 for group in DEGREE_GROUPS}
    group_total = {group: 0 for group in DEGREE_GROUPS}

    for repeat in range(1, 101):
        selected_classes = np.random.choice(
            available_classes,
            size=n_way,
            replace=False,
        )
        supports = []
        queries = []
        support_labels = []
        query_labels = []
        for local_label, class_id in enumerate(selected_classes):
            nodes = class_indices[class_id].copy()
            if len(nodes) > args.k + args.q:
                np.random.shuffle(nodes)
                support_ids = nodes[:args.k]
                query_ids = nodes[args.k:args.k + args.q]
                supports.append(support_ids)
                queries.append(query_ids)
                support_labels.extend([local_label] * len(support_ids))
                query_labels.extend([local_label] * len(query_ids))

        if len(supports) < 2:
            continue

        support_ids = np.concatenate(supports)
        query_ids = np.concatenate(queries)
        support_labels = np.asarray(support_labels)
        query_labels = np.asarray(query_labels)
        classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
        classifier.fit(embeddings[support_ids], support_labels)
        predicted_local = classifier.predict(embeddings[query_ids])
        correct = predicted_local == query_labels
        repeat_accuracies.append(float(correct.mean()))

        for node_id, predicted_label, is_correct in zip(
            query_ids,
            predicted_local,
            correct,
        ):
            group = degree_groups[node_id]
            group_total[group] += 1
            group_correct[group] += int(is_correct)
            prediction_rows.append({
                "dataset": args.dataset_name,
                "setting": task_setting(args),
                "run": run_number,
                "seed": seed,
                "repeat": repeat,
                "query_node_id": int(node_id),
                "node_degree": int(degrees[node_id]),
                "degree_group": group,
                "ground_truth_class": int(labels[node_id]),
                "predicted_class": int(
                    selected_classes[int(predicted_label)]
                ),
                "correct": int(is_correct),
            })

    if not repeat_accuracies:
        raise RuntimeError("No valid meta-test tasks were generated.")

    accuracies = {
        group: (
            100.0 * group_correct[group] / group_total[group]
            if group_total[group] else float("nan")
        )
        for group in DEGREE_GROUPS
    }
    overall = 100.0 * float(np.mean(repeat_accuracies))
    print(
        f"\n最终测试集评估结果: {n_way}-way {args.k}-shot "
        f"(q={args.q}) 准确率: {overall:.2f}%"
    )
    return overall, accuracies, prediction_rows


def build_run_outputs(
    args,
    run_number,
    seed,
    degrees,
    global_groups,
    labels,
    query_counts,
    coverage_rows,
):
    total_queries = int(query_counts.sum())
    sampling_events = len(coverage_rows)
    queries_per_event = total_queries // sampling_events
    class_rows = class_exposure_rows(
        args,
        run_number,
        seed,
        labels,
        query_counts,
    )
    class_tv = 0.5 * sum(
        abs(row["query_share"] - row["node_share"])
        for row in class_rows
    )
    sampling_row = {
        "dataset": args.dataset_name,
        "setting": task_setting(args),
        "run": run_number,
        "seed": seed,
        "num_nodes": len(query_counts),
        "queries_per_epoch": queries_per_event,
        "sampling_events": sampling_events,
        "total_query_selections": total_queries,
        "spearman_degree_query_count": finite_or_blank(
            spearman_correlation(query_counts, degrees)
        ),
        "class_tv": class_tv,
        "final_coverage": coverage_rows[-1]["actual_coverage"],
        "theoretical_final_coverage": coverage_rows[-1][
            "theoretical_coverage"
        ],
    }
    node_rows = (
        {
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "run": run_number,
            "seed": seed,
            "node_id": node_id,
            "node_degree": int(degrees[node_id]),
            "degree_group": global_groups[node_id],
            "ground_truth_class": int(labels[node_id]),
            "query_count": int(query_counts[node_id]),
        }
        for node_id in range(len(query_counts))
    )
    return (
        sampling_row,
        group_exposure_rows(
            args,
            run_number,
            seed,
            degrees,
            global_groups,
            query_counts,
        ),
        class_rows,
        node_rows,
    )


def append_summary_row(rows, args, metric, subgroup, values):
    mean, half, low, high, count = mean_ci95(values)
    rows.append({
        "dataset": args.dataset_name,
        "setting": task_setting(args),
        "metric": metric,
        "subgroup": subgroup,
        "n_runs": count,
        "mean": finite_or_blank(mean),
        "ci95_half_width": finite_or_blank(half),
        "ci95_low": finite_or_blank(low),
        "ci95_high": finite_or_blank(high),
    })


def summarize_experiment(
    args,
    paths,
    sampling_records,
    degree_records,
    class_records,
    coverage_records,
    performance_records,
    lma_records,
):
    summary_rows = []
    for metric in (
        "spearman_degree_query_count",
        "class_tv",
        "final_coverage",
        "theoretical_final_coverage",
    ):
        append_summary_row(
            summary_rows,
            args,
            metric,
            "all",
            [record[metric] for record in sampling_records],
        )

    for group in DEGREE_GROUPS:
        selected = [
            record for record in degree_records
            if record["degree_group"] == group
        ]
        append_summary_row(
            summary_rows,
            args,
            "degree_exposure_ratio",
            group,
            [record["exposure_ratio"] for record in selected],
        )

    for class_id in sorted({row["class_id"] for row in class_records}):
        selected = [
            record for record in class_records
            if record["class_id"] == class_id
        ]
        append_summary_row(
            summary_rows,
            args,
            "class_query_share",
            str(class_id),
            [record["query_share"] for record in selected],
        )
        append_summary_row(
            summary_rows,
            args,
            "class_exposure_ratio",
            str(class_id),
            [record["exposure_ratio"] for record in selected],
        )

    if performance_records:
        performance_metrics = (
            ("overall_accuracy_percent", "all"),
            ("acc_low_percent", "low"),
            ("acc_medium_percent", "medium"),
            ("acc_high_percent", "high"),
            ("delta_low_high_percentage_points", "low-high"),
            ("lma_final20_low", "low"),
            ("lma_final20_medium", "medium"),
            ("lma_final20_high", "high"),
            ("delta_lma_low_high", "low-high"),
        )
        for metric, subgroup in performance_metrics:
            append_summary_row(
                summary_rows,
                args,
                metric,
                subgroup,
                [record[metric] for record in performance_records],
            )

    append_csv(
        paths["summary_95ci.csv"],
        OUTPUT_COLUMNS["summary_95ci.csv"],
        summary_rows,
    )

    coverage_summary = []
    for event in sorted({
        record["sampling_event"] for record in coverage_records
    }):
        selected = [
            record for record in coverage_records
            if record["sampling_event"] == event
        ]
        mean, half, low, high, count = mean_ci95(
            [record["actual_coverage"] for record in selected]
        )
        coverage_summary.append({
            "dataset": args.dataset_name,
            "setting": task_setting(args),
            "sampling_event": event,
            "n_runs": count,
            "mean_actual_coverage": mean,
            "ci95_half_width": half,
            "ci95_low": low,
            "ci95_high": high,
            "theoretical_coverage": selected[0]["theoretical_coverage"],
        })
    append_csv(
        paths["coverage_summary_95ci.csv"],
        OUTPUT_COLUMNS["coverage_summary_95ci.csv"],
        coverage_summary,
    )

    lma_summary = []
    for epoch in sorted({record["epoch"] for record in lma_records}):
        for group in DEGREE_GROUPS:
            selected = [
                record for record in lma_records
                if record["epoch"] == epoch
                and record["degree_group"] == group
            ]
            mean, half, low, high, count = mean_ci95(
                [record["lma"] for record in selected]
            )
            lma_summary.append({
                "dataset": args.dataset_name,
                "setting": task_setting(args),
                "epoch": epoch,
                "degree_group": group,
                "n_runs": count,
                "mean_lma": finite_or_blank(mean),
                "ci95_half_width": finite_or_blank(half),
                "ci95_low": finite_or_blank(low),
                "ci95_high": finite_or_blank(high),
            })
    append_csv(
        paths["lma_summary_95ci.csv"],
        OUTPUT_COLUMNS["lma_summary_95ci.csv"],
        lma_summary,
    )


def write_experiment_metadata(output_dir, args, detailed):
    metadata = {
        "protocol": "minimum query fairness and sparse-node robustness",
        "dataset": args.dataset_name,
        "setting": task_setting(args),
        "candidate_pool": "all graph nodes",
        "degree_definition": (
            "binary undirected degree before CMFC-added self-loops"
        ),
        "training_degree_groups": (
            "global <=33rd, 33rd-67th, >67th percentiles; ties retained"
        ),
        "test_degree_groups": (
            "same thresholds computed independently within each test class"
            if detailed else "not collected for this minimum-protocol dataset"
        ),
        "confidence_interval": (
            "run-level mean +/- 1.96 * sample SD / sqrt(number of runs)"
        ),
        "detailed_sparse_node_analysis": detailed,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "arguments": vars(args),
    }
    with open(
        osp.join(output_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)


def run_query_experiment(args, running_device, output_root):
    num_runs = int(args.num_runs)
    if num_runs <= 0:
        raise ValueError(
            f"num_runs must be a positive integer, got {num_runs}."
        )

    output_dir = experiment_dir(args, output_root)
    paths = initialize_outputs(output_dir)
    prepared_graph = base.prepare_graph(args, running_device)
    degrees = raw_undirected_degrees(prepared_graph.adjacency)
    global_groups = degree_group_labels(degrees)
    labels = prepared_graph.labels_cpu.numpy()
    detailed = args.dataset_name in DETAILED_DATASETS
    write_experiment_metadata(output_dir, args, detailed)

    sampling_records = []
    degree_records = []
    class_records = []
    coverage_records = []
    performance_records = []
    lma_records = []
    base_seed = int(args.seed)
    started = perf_counter()

    for run_index in range(num_runs):
        run_number = run_index + 1
        run_args = copy.deepcopy(args)
        run_args.seed = base_seed + run_index * 1000
        seed = int(run_args.seed)
        print(
            f"\nQuery experiment {run_number}/{num_runs} | "
            f"dataset={args.dataset_name} | seed={seed}"
        )
        masks = base.masks_for_run(prepared_graph, run_args)
        split_description = base.describe_dataset_split(
            prepared_graph,
            masks,
        )
        append_csv(
            paths["run_metadata.csv"],
            OUTPUT_COLUMNS["run_metadata.csv"],
            [{
                "dataset": args.dataset_name,
                "setting": task_setting(args),
                "run": run_number,
                "seed": seed,
                "train_split": split_description[0],
                "validation_split": split_description[1],
                "test_split": split_description[2],
            }],
        )

        model, query_counts, run_coverage, run_lma, lma_final = (
            audited_training_run(
                run_args,
                running_device,
                prepared_graph,
                masks,
                run_number,
                degrees,
                global_groups,
                detailed,
            )
        )
        sampling_row, degree_rows, class_rows, node_rows = (
            build_run_outputs(
                run_args,
                run_number,
                seed,
                degrees,
                global_groups,
                labels,
                query_counts,
                run_coverage,
            )
        )
        sampling_records.append(sampling_row)
        degree_records.extend(degree_rows)
        class_records.extend(class_rows)
        coverage_records.extend(run_coverage)
        lma_records.extend(run_lma)

        append_csv(
            paths["sampling_runs.csv"],
            OUTPUT_COLUMNS["sampling_runs.csv"],
            [sampling_row],
        )
        append_csv(
            paths["degree_exposure_runs.csv"],
            OUTPUT_COLUMNS["degree_exposure_runs.csv"],
            degree_rows,
        )
        append_csv(
            paths["class_exposure_runs.csv"],
            OUTPUT_COLUMNS["class_exposure_runs.csv"],
            class_rows,
        )
        append_csv(
            paths["node_exposure_runs.csv"],
            OUTPUT_COLUMNS["node_exposure_runs.csv"],
            node_rows,
        )
        append_csv(
            paths["coverage_curve.csv"],
            OUTPUT_COLUMNS["coverage_curve.csv"],
            run_coverage,
        )
        append_csv(
            paths["lma_curve.csv"],
            OUTPUT_COLUMNS["lma_curve.csv"],
            run_lma,
        )

        if detailed:
            overall, group_accuracy, prediction_rows = (
                evaluate_with_degree_predictions(
                    model,
                    prepared_graph,
                    masks.test,
                    run_args,
                    run_number,
                    seed,
                    degrees,
                )
            )
            append_csv(
                paths["test_predictions.csv"],
                OUTPUT_COLUMNS["test_predictions.csv"],
                prediction_rows,
            )
            performance_row = {
                "dataset": args.dataset_name,
                "setting": task_setting(args),
                "run": run_number,
                "seed": seed,
                "overall_accuracy_percent": overall,
                "acc_low_percent": finite_or_blank(
                    group_accuracy["low"]
                ),
                "acc_medium_percent": finite_or_blank(
                    group_accuracy["medium"]
                ),
                "acc_high_percent": finite_or_blank(
                    group_accuracy["high"]
                ),
                "delta_low_high_percentage_points": finite_or_blank(
                    group_accuracy["low"] - group_accuracy["high"]
                ),
                "lma_final20_low": finite_or_blank(lma_final["low"]),
                "lma_final20_medium": finite_or_blank(
                    lma_final["medium"]
                ),
                "lma_final20_high": finite_or_blank(lma_final["high"]),
                "delta_lma_low_high": finite_or_blank(
                    lma_final["low"] - lma_final["high"]
                ),
            }
            performance_records.append(performance_row)
            append_csv(
                paths["performance_runs.csv"],
                OUTPUT_COLUMNS["performance_runs.csv"],
                [performance_row],
            )
        else:
            base.evaluate_on_test_set(
                model,
                prepared_graph,
                masks.test,
                run_args,
                "最终测试集",
            )

    summarize_experiment(
        args,
        paths,
        sampling_records,
        degree_records,
        class_records,
        coverage_records,
        performance_records,
        lma_records,
    )
    elapsed = perf_counter() - started
    print(
        f"\nQuery audit completed in {elapsed:.1f}s.\n"
        f"Results saved to: {output_dir}"
    )
    return output_dir


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    query_options, remaining = parse_query_options(argv)

    if query_options.minimum_suite:
        for dataset_name, n_way, k_shot in MINIMUM_SUITE:
            args = load_args(
                remaining,
                overrides=(
                    "--dataset-name", dataset_name,
                    "--n", str(n_way),
                    "--k", str(k_shot),
                ),
            )
            pprint_args(args)
            running_device = base.resolve_configured_device(args)
            print(f"Using device: {running_device}")
            run_query_experiment(
                args,
                running_device,
                query_options.output_root,
            )
        return

    args = load_args(remaining)
    pprint_args(args)
    running_device = base.resolve_configured_device(args)
    print(f"Using device: {running_device}")
    run_query_experiment(args, running_device, query_options.output_root)


if __name__ == "__main__":
    main()
