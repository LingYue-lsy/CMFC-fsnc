"""Train CMFC once and record the top-k label match rate per epoch.

Example:
    python train_label.py --dataset-name Cora --n 2 --k 5 --seed 1
"""

import csv
import os
import os.path as osp
import random
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

import train as base
from args_utils import get_args, pprint_args
from models import CMFC_Fast


LABEL_COLUMNS = [
    "epoch",
    "target_node_count",
    "topk",
    "matching_pairs",
    "total_pairs",
    "label_match_probability",
]


def label_csv_path(args, output_root=None):
    """Return the per-seed label-match CSV path."""
    if output_root is None:
        output_root = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "results",
            "label",
        )
    filename = f"{args.n}-way {args.k}-shot seed-{int(args.seed)}.csv"
    return osp.join(
        osp.abspath(output_root),
        str(args.dataset_name),
        str(args.support_selection),
        filename,
    )


def label_match_metrics(labels, target_indices, topk_indices):
    """Calculate aggregate label matches for the selected target/support pairs."""
    if topk_indices.ndim != 2:
        raise ValueError(
            "topk_indices must be a two-dimensional tensor, got "
            f"shape={tuple(topk_indices.shape)}."
        )
    if target_indices.ndim != 1:
        raise ValueError(
            "target_indices must be a one-dimensional tensor, got "
            f"shape={tuple(target_indices.shape)}."
        )
    if topk_indices.size(0) != target_indices.numel():
        raise ValueError(
            "topk_indices must contain one row per target node, got "
            f"{topk_indices.size(0)} rows for {target_indices.numel()} targets."
        )

    target_labels = labels[target_indices].unsqueeze(1)
    selected_labels = labels[topk_indices]
    matching_pairs = int((selected_labels == target_labels).sum().item())
    total_pairs = int(topk_indices.numel())
    probability = matching_pairs / total_pairs if total_pairs else 0.0
    return matching_pairs, total_pairs, probability


def initialize_label_csv(csv_path):
    """Create or replace a label-match CSV with its header."""
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=LABEL_COLUMNS).writeheader()


def append_label_row(csv_path, row):
    """Append one epoch's aggregate label-match metrics."""
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=LABEL_COLUMNS).writerow(row)


def run_label_training(
    args,
    running_device,
    prepared_graph=None,
    masks=None,
    csv_path=None,
):
    """Run one seeded CMFC training and write one label-match row per epoch."""
    if int(args.k) <= 0:
        raise ValueError(f"k must be a positive integer, got {args.k}.")
    if int(args.m) <= 0:
        raise ValueError(f"m must be a positive integer, got {args.m}.")

    seed = int(args.seed)
    # args_utils currently leaves command-line seed overrides as strings.
    # Normalize it here before dataset split helpers consume the value.
    args.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    running_device = base.resolve_device(running_device)
    if prepared_graph is None:
        prepared_graph = base.prepare_graph(args, running_device)
    if prepared_graph.dataset_name != args.dataset_name:
        raise ValueError(
            "Prepared graph dataset does not match args.dataset_name: "
            f"{prepared_graph.dataset_name!r} != {args.dataset_name!r}."
        )
    if masks is None:
        masks = base.masks_for_run(prepared_graph, args)

    csv_path = csv_path or label_csv_path(args)
    initialize_label_csv(csv_path)

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
    started = perf_counter()

    for epoch in tqdm(base.training_epoch_range(args), desc="label match"):
        model.train()
        optimiser.zero_grad()
        h_a, h_p, h_a2 = model(
            prepared_graph.features,
            prepared_graph.adjacency,
        )

        target_indices_np = np.random.choice(
            prepared_graph.num_nodes,
            size=num_task_node,
            replace=False,
        )
        target_indices = torch.as_tensor(
            target_indices_np,
            dtype=torch.long,
            device=running_device,
        )
        negative_indices = torch.as_tensor(
            np.stack([
                np.random.permutation(num_task_node)
                for _ in range(num_neg)
            ]),
            dtype=torch.long,
            device=running_device,
        )

        target_embeddings = h_a[target_indices]
        topk_indices = base.select_support_indices(
            target_embeddings,
            h_a2,
            args.k,
            args.support_selection,
        )
        with torch.no_grad():
            matching_pairs, total_pairs, probability = label_match_metrics(
                prepared_graph.labels,
                target_indices,
                topk_indices,
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
                target_embeddings,
                h_p,
                neighbor_embeddings,
                topk_indices,
                negative_indices,
            )
        )
        support_loss, neighbor_loss, mask_loss = (
            base.compute_batched_margin_terms(
                support_distances,
                neighbor_distances,
                negative_distances,
                args.margin1,
                mask_margin,
            )
        )
        loss_mar = support_loss / (num_task_node * args.k) * 100
        loss_mar_1 = neighbor_loss / (num_task_node * args.k) * 100
        mask_margin_n = (
            mask_loss / (num_task_node * args.k * num_neg) * 100
        )
        loss = (
            loss_mar * args.w_loss1
            + loss_mar_1 * args.w_loss2
            + mask_margin_n * args.w_loss3
        )
        loss.backward()
        optimiser.step()

        append_label_row(csv_path, {
            "epoch": epoch + 1,
            "target_node_count": num_task_node,
            "topk": int(args.k),
            "matching_pairs": matching_pairs,
            "total_pairs": total_pairs,
            "label_match_probability": f"{probability:.12g}",
        })

        if args.save_model:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                osp.join(
                    args.checkpoint_dir,
                    f"{args.dataset_name}_weights.pth",
                ),
            )

    accuracy = base.evaluate_on_test_set(
        model,
        prepared_graph,
        masks.test,
        args,
        "最终测试集",
    )
    elapsed = perf_counter() - started
    print(f"total time {elapsed:.2f}s")
    print(f"Label match results saved to: {csv_path}")
    return accuracy, csv_path


def main():
    args = get_args(
        model_name="CMFC",
        dataset_class="Planetoid",
        dataset_name=None,
        custom_key="classification",
    )
    pprint_args(args)
    running_device = base.resolve_configured_device(args)
    print(f"Using device: {running_device}")
    accuracy, _ = run_label_training(args, running_device)
    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
