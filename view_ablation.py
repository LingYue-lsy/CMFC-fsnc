import argparse
import copy
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from args_utils import get_args, pprint_args
from models import CMFC_Fast
from train import (
    aggregate_sampled_neighbors,
    compute_batched_distances,
    compute_batched_margin_terms,
    get_cached_prepared_graph,
    masks_for_run,
    resolve_configured_device,
    select_support_indices,
    training_epoch_range,
)


VIEW_MODES = ('full', 'no_graph', 'no_neighbor', 'mlp_only')
VIEW_MODE_CONFIG = {
    'full': {'use_graph': True, 'use_neighbor': True},
    'no_graph': {'use_graph': False, 'use_neighbor': True},
    'no_neighbor': {'use_graph': True, 'use_neighbor': False},
    'mlp_only': {'use_graph': False, 'use_neighbor': False},
}
VIEW_PAIRS = (
    ('MLP-Graph', 'mlp', 'graph'),
    ('MLP-Neighbor', 'mlp', 'neighbor'),
    ('Graph-Neighbor', 'graph', 'neighbor'),
)


def parse_experiment_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--view-modes',
        '--view_modes',
        dest='view_modes',
        nargs='+',
        choices=VIEW_MODES,
        default=list(VIEW_MODES),
    )
    parser.add_argument(
        '--ablation-output-dir',
        '--ablation_output_dir',
        dest='ablation_output_dir',
        default='results/ablation',
    )
    experiment_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = get_args(
            model_name='CMFC',
            dataset_class='Planetoid',
            dataset_name=None,
            custom_key='classification',
        )
    finally:
        sys.argv = original_argv

    experiment_args.view_modes = list(dict.fromkeys(experiment_args.view_modes))
    return args, experiment_args


def set_run_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forward_variant(model, features, adjacency, use_graph):
    if use_graph:
        return model(features, adjacency)

    dropped_features = F.dropout(
        features,
        model.dropout,
        training=model.training,
    )
    online_embeddings = model.MLP(dropped_features)
    if model.training:
        model._update_ema_weights()
    with torch.no_grad():
        mlp_embeddings = model.MLP_ema(dropped_features)
    return online_embeddings, None, mlp_embeddings


def compute_support_only_terms(
    task_embeddings,
    support_embeddings,
    topk_indices,
    negative_indices,
    margin,
    mask_margin,
):
    task_for_support = task_embeddings.unsqueeze(1).expand(
        -1,
        topk_indices.size(1),
        -1,
    )
    support_distances = F.pairwise_distance(
        task_for_support,
        support_embeddings[topk_indices],
    )
    task_for_negative = task_embeddings.unsqueeze(0).expand(
        negative_indices.size(0),
        -1,
        -1,
    )
    negative_distances = F.pairwise_distance(
        task_for_negative,
        task_embeddings[negative_indices],
    )

    support_expanded = support_distances.unsqueeze(0).expand(
        negative_distances.size(0),
        -1,
        -1,
    )
    negative_expanded = negative_distances.unsqueeze(-1).expand_as(
        support_expanded
    )
    labels = -torch.ones_like(support_expanded)
    support_margin = F.margin_ranking_loss(
        support_expanded,
        negative_expanded,
        labels,
        margin=margin,
        reduction='none',
    )
    mask_values = torch.clamp(
        negative_expanded - support_expanded.detach() - mask_margin,
        min=0,
    )
    return support_margin.mean(dim=1).sum(), mask_values.sum()


def get_evaluation_embeddings(model, prepared_graph, use_graph):
    model.eval()
    with torch.no_grad():
        if use_graph:
            mlp_embeddings, graph_embeddings, _ = model.embed(
                prepared_graph.features,
                prepared_graph.adjacency,
            )
            del mlp_embeddings
            embeddings = graph_embeddings
        else:
            embeddings = model.MLP_ema(prepared_graph.features)
        embeddings = embeddings / embeddings.norm(dim=1)[:, None]
    return embeddings.cpu().numpy()


def evaluate_variant(model, prepared_graph, test_mask, args, use_graph):
    embeddings = get_evaluation_embeddings(
        model,
        prepared_graph,
        use_graph,
    )
    eval_indices = torch.nonzero(
        test_mask,
        as_tuple=False,
    ).view(-1).cpu().numpy()
    labels = prepared_graph.labels_cpu.numpy()
    eval_labels = labels[eval_indices]
    available_classes = np.unique(eval_labels)
    n_way = min(args.n, len(available_classes))
    class_indices_by_label = {
        class_idx: eval_indices[eval_labels == class_idx]
        for class_idx in available_classes
    }

    scores = []
    for _ in range(100):
        selected_classes = np.random.choice(
            available_classes,
            size=n_way,
            replace=False,
        )
        supports = []
        queries = []
        support_labels = []
        query_labels = []

        for label_idx, class_idx in enumerate(selected_classes):
            class_indices = class_indices_by_label[class_idx].copy()
            if len(class_indices) > args.k + args.q:
                np.random.shuffle(class_indices)
                support_idx = class_indices[:args.k]
                query_idx = class_indices[args.k:args.k + args.q]
                supports.append(support_idx)
                queries.append(query_idx)
                support_labels.extend([label_idx] * len(support_idx))
                query_labels.extend([label_idx] * len(query_idx))

        if len(supports) < 2:
            continue

        support_idx = np.concatenate(supports)
        query_idx = np.concatenate(queries)
        classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        classifier.fit(
            embeddings[support_idx],
            np.asarray(support_labels),
        )
        scores.append(classifier.score(
            embeddings[query_idx],
            np.asarray(query_labels),
        ))

    if not scores:
        raise RuntimeError('Evaluation produced no valid few-shot episodes.')
    mean_accuracy = float(np.mean(scores) * 100)
    episode_interval = float(
        1.96 * np.std(scores) / np.sqrt(len(scores)) * 100
    )
    print(
        f"Evaluation: {n_way}-way {args.k}-shot (q={args.q}) "
        f"accuracy={mean_accuracy:.2f}±{episode_interval:.2f}%"
    )
    return mean_accuracy


def linear_cka(first, second):
    first = first.detach().cpu().double()
    second = second.detach().cpu().double()
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)

    cross_covariance = first.T @ second
    first_covariance = first.T @ first
    second_covariance = second.T @ second
    numerator = cross_covariance.square().sum()
    denominator = torch.sqrt(
        first_covariance.square().sum()
        * second_covariance.square().sum()
    )
    if denominator.item() == 0:
        raise ValueError('CKA is undefined for a constant representation.')
    return float(torch.clamp(numerator / denominator, 0.0, 1.0))


def mean_cosine_similarity(first, second):
    return float(F.cosine_similarity(
        first.detach().cpu(),
        second.detach().cpu(),
        dim=1,
    ).mean())


def measure_full_view_complementarity(
    model,
    prepared_graph,
    args,
    final_epoch,
):
    model.eval()
    with torch.no_grad():
        mlp_embeddings, graph_embeddings, _ = model.embed(
            prepared_graph.features,
            prepared_graph.adjacency,
        )
        all_nodes = torch.arange(
            prepared_graph.num_nodes,
            device=prepared_graph.features.device,
        ).unsqueeze(1)
        neighbor_embeddings = aggregate_sampled_neighbors(
            mlp_embeddings,
            all_nodes,
            prepared_graph.neighbor_table,
            final_epoch,
            args.m,
        ).squeeze(1)

    views = {
        'mlp': mlp_embeddings,
        'graph': graph_embeddings,
        'neighbor': neighbor_embeddings,
    }
    return {
        pair_name: {
            'linear_cka': linear_cka(views[first], views[second]),
            'cosine_similarity': mean_cosine_similarity(
                views[first],
                views[second],
            ),
        }
        for pair_name, first, second in VIEW_PAIRS
    }


def save_variant_checkpoint(model, args, mode, run_id):
    if not args.save_model:
        return
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{args.dataset_name}_{mode}_run{run_id}_weights.pth"
    torch.save(
        model.state_dict(),
        os.path.join(args.checkpoint_dir, checkpoint_name),
    )


def train_variant(args, prepared_graph, mode, run_id):
    config = VIEW_MODE_CONFIG[mode]
    use_graph = config['use_graph']
    use_neighbor = config['use_neighbor']
    set_run_seed(int(args.seed))
    masks = masks_for_run(prepared_graph, args)
    device = prepared_graph.features.device

    model = CMFC_Fast(
        prepared_graph.num_features,
        cfg=args.cfg,
        dropout=args.dropout,
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_negative = args.NN
    margin = args.margin1
    mask_margin = args.margin1 + args.margin2
    final_epoch = None

    for epoch in tqdm(training_epoch_range(args), desc=mode):
        final_epoch = epoch
        model.train()
        optimiser.zero_grad()
        online, graph, mlp = forward_variant(
            model,
            prepared_graph.features,
            prepared_graph.adjacency,
            use_graph,
        )
        num_task_nodes = min(
            args.n * args.task_num,
            prepared_graph.num_nodes,
        )
        task_indices = torch.as_tensor(
            np.random.choice(
                prepared_graph.num_nodes,
                size=num_task_nodes,
                replace=False,
            ),
            dtype=torch.long,
            device=device,
        )
        negative_indices = torch.as_tensor(
            np.stack([
                np.random.permutation(num_task_nodes)
                for _ in range(num_negative)
            ]),
            dtype=torch.long,
            device=device,
        )
        task_embeddings = online[task_indices]
        topk_indices = select_support_indices(
            task_embeddings,
            mlp,
            args.k,
            args.support_selection,
        )
        support_embeddings = graph if use_graph else mlp

        if use_neighbor:
            neighbor_embeddings = aggregate_sampled_neighbors(
                mlp,
                topk_indices,
                prepared_graph.neighbor_table,
                epoch,
                args.m,
            )
            support_distances, neighbor_distances, negative_distances = (
                compute_batched_distances(
                    task_embeddings,
                    support_embeddings,
                    neighbor_embeddings,
                    topk_indices,
                    negative_indices,
                )
            )
            support_term, neighbor_term, mask_term = (
                compute_batched_margin_terms(
                    support_distances,
                    neighbor_distances,
                    negative_distances,
                    margin,
                    mask_margin,
                )
            )
        else:
            support_term, mask_term = compute_support_only_terms(
                task_embeddings,
                support_embeddings,
                topk_indices,
                negative_indices,
                margin,
                mask_margin,
            )
            neighbor_term = None

        support_loss = support_term / (num_task_nodes * args.k) * 100
        mask_loss = (
            mask_term
            / (num_task_nodes * args.k * num_negative)
            * 100
        )
        loss = support_loss * args.w_loss1 + mask_loss * args.w_loss3
        if neighbor_term is not None:
            neighbor_loss = neighbor_term / (num_task_nodes * args.k) * 100
            loss = loss + neighbor_loss * args.w_loss2

        loss.backward()
        optimiser.step()
        save_variant_checkpoint(model, args, mode, run_id)

    accuracy = evaluate_variant(
        model,
        prepared_graph,
        masks.test,
        args,
        use_graph,
    )
    return model, accuracy, final_epoch


def confidence_summary(values, digits):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    interval = float(1.96 * np.std(values) / np.sqrt(len(values)))
    return f"{mean:.{digits}f}±{interval:.{digits}f}"


def write_ablation_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['variant', 'accuracy'])
        writer.writerows(rows)


def write_complementarity_csv(path, metrics):
    with open(path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'view_pair',
            'linear_cka',
            'cosine_similarity',
        ])
        for pair_name, _, _ in VIEW_PAIRS:
            pair_metrics = metrics[pair_name]
            writer.writerow([
                pair_name,
                confidence_summary(pair_metrics['linear_cka'], 4),
                confidence_summary(pair_metrics['cosine_similarity'], 4),
            ])


def run_ablation(args, experiment_args):
    if int(args.num_runs) <= 0:
        raise ValueError(f"num_runs must be positive, got {args.num_runs}.")
    if int(args.m) <= 0:
        raise ValueError(f"m must be positive, got {args.m}.")

    device = resolve_configured_device(args)
    graph_cache = {}
    prepared_graph = get_cached_prepared_graph(args, device, graph_cache)
    os.makedirs(experiment_args.ablation_output_dir, exist_ok=True)
    filename_prefix = (
        f"{args.dataset_name}_{args.n}-way_{args.k}-shot_view"
    )
    ablation_path = os.path.join(
        experiment_args.ablation_output_dir,
        f"{filename_prefix}_ablation.csv",
    )
    complementarity_path = os.path.join(
        experiment_args.ablation_output_dir,
        f"{filename_prefix}_complementarity.csv",
    )
    ablation_rows = []
    complementarity = {
        pair_name: {'linear_cka': [], 'cosine_similarity': []}
        for pair_name, _, _ in VIEW_PAIRS
    }
    write_ablation_csv(ablation_path, ablation_rows)

    base_seed = int(args.seed)
    for mode in experiment_args.view_modes:
        print(f"\n{'=' * 30}\nView ablation: {mode}\n{'=' * 30}")
        accuracies = []
        for run_id in range(int(args.num_runs)):
            run_args = copy.deepcopy(args)
            run_args.seed = base_seed + run_id * 1000
            print(
                f"Run {run_id + 1}/{args.num_runs} | "
                f"Seed: {run_args.seed}"
            )
            model, accuracy, final_epoch = train_variant(
                run_args,
                prepared_graph,
                mode,
                run_id,
            )
            accuracies.append(accuracy)

            if mode == 'full':
                run_metrics = measure_full_view_complementarity(
                    model,
                    prepared_graph,
                    run_args,
                    final_epoch,
                )
                for pair_name, values in run_metrics.items():
                    complementarity[pair_name]['linear_cka'].append(
                        values['linear_cka']
                    )
                    complementarity[pair_name]['cosine_similarity'].append(
                        values['cosine_similarity']
                    )
            del model

        formatted_accuracy = confidence_summary(accuracies, 2)
        ablation_rows.append([mode, formatted_accuracy])
        write_ablation_csv(ablation_path, ablation_rows)
        print(f"{mode}: {formatted_accuracy}")

        if mode == 'full':
            write_complementarity_csv(
                complementarity_path,
                complementarity,
            )

    print(f"\nAblation results saved to: {ablation_path}")
    if 'full' in experiment_args.view_modes:
        print(
            "Complementarity results saved to: "
            f"{complementarity_path}"
        )
    return ablation_rows, complementarity


def main():
    args, experiment_args = parse_experiment_args()
    pprint_args(args)
    print(f"View modes: {experiment_args.view_modes}")
    print(f"Ablation output: {experiment_args.ablation_output_dir}")
    run_ablation(args, experiment_args)


if __name__ == '__main__':
    main()
