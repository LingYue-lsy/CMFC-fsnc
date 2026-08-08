"""Train CMFC once and record all loss terms after every epoch.

The output is written to::

    results/loss/<dataset>/<n>-way <k>-shot seed-<seed>.csv

Each row contains the support margin loss (``Ls``), neighbor margin loss
(``Ln``), negative-mask margin loss (``Lu``), and their weighted sum (``L``).

Example::

    python train_loss.py --dataset-name Cora --n 2 --k 5 --seed 1
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


LOSS_COLUMNS = ["epoch", "Ls", "Ln", "Lu", "L"]


def loss_csv_path(args, output_root=None):
    """Return the per-seed loss CSV path.

    ``output_root`` is injectable for tests and for callers that need a
    different output location.  By default paths are rooted at this project.
    """
    if output_root is None:
        output_root = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "results",
            "loss",
        )
    filename = f"{args.n}-way {args.k}-shot seed-{int(args.seed)}.csv"
    return osp.join(osp.abspath(output_root), str(args.dataset_name), filename)


def initialize_loss_csv(csv_path):
    """Create or replace a loss CSV with its header."""
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=LOSS_COLUMNS).writeheader()


def append_loss_row(csv_path, row):
    """Append one epoch's loss values to ``csv_path``."""
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=LOSS_COLUMNS).writerow(row)


def run_loss_training(
    args,
    running_device,
    prepared_graph=None,
    masks=None,
    csv_path=None,
):
    """Run one seeded CMFC training and write one row per training epoch."""
    if int(args.n) <= 0:
        raise ValueError(f"n must be a positive integer, got {args.n}.")
    if int(args.k) <= 0:
        raise ValueError(f"k must be a positive integer, got {args.k}.")
    if int(args.m) <= 0:
        raise ValueError(f"m must be a positive integer, got {args.m}.")
    if int(args.NN) <= 0:
        raise ValueError(f"NN must be a positive integer, got {args.NN}.")

    seed = int(args.seed)
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

    csv_path = csv_path or loss_csv_path(args)
    initialize_loss_csv(csv_path)

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

    num_neg = int(args.NN)
    margin = args.margin1
    mask_margin = margin + args.margin2
    num_task_node = min(int(args.n) * int(args.task_num), prepared_graph.num_nodes)
    started = perf_counter()

    for epoch in tqdm(base.training_epoch_range(args), desc="loss"):
        model.train()
        optimiser.zero_grad()
        h_a, h_p, h_a2 = model(
            prepared_graph.features,
            prepared_graph.adjacency,
        )

        task_indices = torch.as_tensor(
            np.random.choice(
                prepared_graph.num_nodes,
                size=num_task_node,
                replace=False,
            ),
            dtype=torch.long,
            device=running_device,
        )
        negative_indices = torch.as_tensor(
            np.stack([np.random.permutation(num_task_node) for _ in range(num_neg)]),
            dtype=torch.long,
            device=running_device,
        )

        task_embeddings = h_a[task_indices]
        topk_indices = base.select_support_indices(
            task_embeddings,
            h_a2,
            int(args.k),
            args.support_selection,
        )
        neighbor_embeddings = base.aggregate_sampled_neighbors(
            h_a2,
            topk_indices,
            prepared_graph.neighbor_table,
            epoch,
            int(args.m),
        )
        support_distances, neighbor_distances, negative_distances = (
            base.compute_batched_distances(
                task_embeddings,
                h_p,
                neighbor_embeddings,
                topk_indices,
                negative_indices,
            )
        )
        support_loss, neighbor_loss, mask_loss = base.compute_batched_margin_terms(
            support_distances,
            neighbor_distances,
            negative_distances,
            margin,
            mask_margin,
        )

        # Keep these normalizations identical to train.py so the recorded
        # values are directly comparable with the training objective.
        loss_mar = support_loss / (num_task_node * int(args.k))
        loss_mar_1 = neighbor_loss / (num_task_node * int(args.k))
        mask_margin_N = mask_loss / (num_task_node * int(args.k) * num_neg)
        loss = (
            loss_mar * args.w_loss1
            + loss_mar_1 * args.w_loss2
            + mask_margin_N * args.w_loss3
        )
        loss.backward()
        optimiser.step()

        append_loss_row(csv_path, {
            "epoch": epoch + 1,
            "Ls": f"{loss_mar.detach().item():.12g}",
            "Ln": f"{loss_mar_1.detach().item():.12g}",
            "Lu": f"{mask_margin_N.detach().item():.12g}",
            "L": f"{loss.detach().item():.12g}",
        })

        if args.save_model:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                osp.join(args.checkpoint_dir, f"{args.dataset_name}_weights.pth"),
            )

    accuracy = base.evaluate_on_test_set(
        model,
        prepared_graph,
        masks.test,
        args,
        "test set",
    )
    elapsed = perf_counter() - started
    print(f"total time {elapsed:.2f}s")
    print(f"Loss results saved to: {csv_path}")
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
    accuracy, _ = run_loss_training(args, running_device)
    if accuracy is not None:
        print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
