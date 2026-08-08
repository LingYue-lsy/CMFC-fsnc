import argparse
import copy
import csv
from dataclasses import replace
import os
import os.path as osp
import sys

import numpy as np
import torch

from args_utils import get_args, pprint_args
from train import (
    PreparedGraph,
    build_neighbor_table,
    build_normalized_sparse_adjacency,
    get_cached_prepared_graph,
    resolve_configured_device,
    run_CMFC,
)


FEATURE_MASK_SEED_OFFSET = 104729


def parse_perturbation_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--p',
        type=float,
        default=None,
        help='probability of dropping each undirected non-self-loop edge',
    )
    parser.add_argument(
        '--feature-mask-ratio',
        '--feature_mask_ratio',
        dest='feature_mask_ratio',
        type=float,
        default=None,
        help='probability of masking each feature dimension',
    )
    perturbation_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return perturbation_args


def validate_ratio(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must satisfy 0 <= value <= 1, got {value}.')


def validate_controlled_perturbation(args):
    validate_ratio('p', args.p)
    validate_ratio('feature_mask_ratio', args.feature_mask_ratio)
    if args.p > 0.0 and args.feature_mask_ratio > 0.0:
        raise ValueError(
            'Controlled ablation requires only one active perturbation: '
            'set either p or feature_mask_ratio to 0.'
        )


def extract_undirected_edges(adjacency):
    adjacency = adjacency.coalesce()
    row, col = adjacency.indices()
    non_self_canonical = row < col
    edges = torch.stack(
        [row[non_self_canonical], col[non_self_canonical]],
        dim=0,
    )
    if edges.size(1) == 0:
        raise ValueError('The input graph contains no non-self-loop edges.')
    return edges


def perturb_prepared_graph(
    prepared_graph: PreparedGraph,
    undirected_edges: torch.Tensor,
    p: float,
    seed: int,
):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    edge_scores = torch.rand(
        undirected_edges.size(1),
        generator=generator,
        device='cpu',
    )
    keep_mask = (edge_scores >= p).to(undirected_edges.device)
    retained_edges = undirected_edges[:, keep_mask]

    directed_edges = torch.cat(
        [retained_edges, retained_edges.flip(0)],
        dim=1,
    )
    perturbed_adjacency = build_normalized_sparse_adjacency(
        directed_edges,
        prepared_graph.num_nodes,
    ).to(prepared_graph.features.device).coalesce()
    perturbed_neighbor_table = build_neighbor_table(perturbed_adjacency)

    original_edge_count = undirected_edges.size(1)
    retained_edge_count = retained_edges.size(1)
    edge_retention = retained_edge_count / original_edge_count
    perturbed_graph = replace(
        prepared_graph,
        adjacency=perturbed_adjacency,
        neighbor_table=perturbed_neighbor_table,
    )
    return (
        perturbed_graph,
        edge_retention,
        retained_edge_count,
        original_edge_count,
    )


def mask_prepared_features(
    prepared_graph: PreparedGraph,
    feature_mask_ratio: float,
    seed: int,
):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + FEATURE_MASK_SEED_OFFSET)
    feature_scores = torch.rand(
        prepared_graph.num_features,
        generator=generator,
        device='cpu',
    )
    keep_mask = feature_scores >= feature_mask_ratio
    device_keep_mask = keep_mask.to(
        device=prepared_graph.features.device,
        dtype=prepared_graph.features.dtype,
    )
    masked_features = (
        prepared_graph.features * device_keep_mask.unsqueeze(0)
    )

    original_feature_count = prepared_graph.num_features
    retained_feature_count = int(keep_mask.sum().item())
    feature_retention = retained_feature_count / original_feature_count
    masked_graph = replace(
        prepared_graph,
        features=masked_features,
    )
    return (
        masked_graph,
        feature_retention,
        retained_feature_count,
        original_feature_count,
    )


def run_repeated_perturbation_experiment(
    args,
    running_device,
    prepared_graph=None,
    undirected_edges=None,
):
    validate_controlled_perturbation(args)
    num_runs = int(args.num_runs)
    if num_runs <= 0:
        raise ValueError(f'num_runs must be positive, got {num_runs}.')

    if prepared_graph is None:
        prepared_graph = get_cached_prepared_graph(
            args,
            running_device,
            {},
        )
    if undirected_edges is None:
        undirected_edges = extract_undirected_edges(
            prepared_graph.adjacency
        )
    base_seed = int(args.seed)
    run_results = []
    edge_retentions = []
    feature_retentions = []

    for run_id in range(num_runs):
        run_args = copy.deepcopy(args)
        run_seed = base_seed + run_id * 1000
        run_args.seed = run_seed
        (
            perturbed_graph,
            edge_retention,
            retained_edge_count,
            original_edge_count,
        ) = perturb_prepared_graph(
            prepared_graph,
            undirected_edges,
            args.p,
            run_seed,
        )
        (
            perturbed_graph,
            feature_retention,
            retained_feature_count,
            original_feature_count,
        ) = mask_prepared_features(
            perturbed_graph,
            args.feature_mask_ratio,
            run_seed,
        )

        print(
            f'\nExperiment {run_id + 1}/{num_runs} | Seed: {run_seed} | '
            f'p={args.p:.4f} | '
            f'feature_mask_ratio={args.feature_mask_ratio:.4f}'
        )
        print(
            f'Undirected edges retained: {retained_edge_count}/'
            f'{original_edge_count} ({edge_retention:.6f})'
        )
        print(
            f'Feature dimensions retained: {retained_feature_count}/'
            f'{original_feature_count} ({feature_retention:.6f})'
        )

        run_result = run_CMFC(
            run_args,
            gpu_id=running_device,
            run_id=run_id,
            prepared_graph=perturbed_graph,
        )
        if run_result is None:
            raise RuntimeError(
                f'Experiment {run_id + 1}/{num_runs} returned no accuracy.'
            )
        run_results.append(run_result)
        edge_retentions.append(edge_retention)
        feature_retentions.append(feature_retention)

    mean_accuracy = float(np.mean(run_results))
    confidence_interval = float(
        1.96 * np.std(run_results) / np.sqrt(len(run_results))
    )
    mean_edge_retention = float(np.mean(edge_retentions))
    edge_retention_interval = float(
        1.96 * np.std(edge_retentions) / np.sqrt(len(edge_retentions))
    )
    mean_feature_retention = float(np.mean(feature_retentions))
    feature_retention_interval = float(
        1.96
        * np.std(feature_retentions)
        / np.sqrt(len(feature_retentions))
    )
    return (
        mean_accuracy,
        confidence_interval,
        mean_edge_retention,
        edge_retention_interval,
        mean_feature_retention,
        feature_retention_interval,
    )


def save_perturbation_sweep_results(csv_path, parameter_name, rows):
    retention_column = {
        'p': 'actual_edge_retention',
        'feature_mask_ratio': 'actual_attribute_retention',
    }[parameter_name]
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([parameter_name, retention_column, 'accuracy'])
        writer.writerows(rows)


def run_perturbation_sweep(args, running_device):
    parameter_name = args.sweep_parameter
    supported_parameters = {'p', 'feature_mask_ratio'}
    if parameter_name not in supported_parameters:
        raise ValueError(
            'perturbation_ablation.py supports sweep_parameter values '
            f'{sorted(supported_parameters)}, got {parameter_name!r}.'
        )
    if not args.sweep_values:
        raise ValueError('sweep_values must contain at least one value.')

    sweep_values = [float(value) for value in args.sweep_values]
    for value in sweep_values:
        validate_ratio(parameter_name, value)

    fixed_parameter = (
        'feature_mask_ratio' if parameter_name == 'p' else 'p'
    )
    if getattr(args, fixed_parameter) != 0.0:
        raise ValueError(
            f'{fixed_parameter} must be 0 when sweeping {parameter_name}.'
        )

    os.makedirs(args.sweep_output_dir, exist_ok=True)
    csv_name = (
        f'{args.dataset_name}_{args.n}-way_{args.k}-shot_'
        f'{parameter_name}.csv'
    )
    csv_path = osp.join(args.sweep_output_dir, csv_name)
    summary_rows = []
    save_perturbation_sweep_results(
        csv_path,
        parameter_name,
        summary_rows,
    )

    prepared_graph = get_cached_prepared_graph(
        args,
        running_device,
        {},
    )
    undirected_edges = extract_undirected_edges(
        prepared_graph.adjacency
    )

    for value in sweep_values:
        sweep_args = copy.deepcopy(args)
        setattr(sweep_args, parameter_name, value)
        validate_controlled_perturbation(sweep_args)
        print(
            f"\n{'=' * 30}\n"
            f'Perturbation sweep: {parameter_name}={value}\n'
            f"{'=' * 30}"
        )

        (
            mean_accuracy,
            confidence_interval,
            mean_edge_retention,
            edge_retention_interval,
            mean_feature_retention,
            feature_retention_interval,
        ) = run_repeated_perturbation_experiment(
            sweep_args,
            running_device,
            prepared_graph,
            undirected_edges,
        )
        if parameter_name == 'p':
            mean_retention = mean_edge_retention
            retention_interval = edge_retention_interval
        else:
            mean_retention = mean_feature_retention
            retention_interval = feature_retention_interval
        formatted_retention = (
            f'{mean_retention:.6f}±{retention_interval:.6f}'
        )
        formatted_accuracy = (
            f'{mean_accuracy:.2f}±{confidence_interval:.2f}'
        )
        summary_rows.append([
            f'{value:g}',
            formatted_retention,
            formatted_accuracy,
        ])
        save_perturbation_sweep_results(
            csv_path,
            parameter_name,
            summary_rows,
        )
        print(
            f'{parameter_name}={value:g}: '
            f'retention={formatted_retention}, '
            f'accuracy={formatted_accuracy}'
        )

    print(f'\nPerturbation sweep results saved to: {csv_path}')
    return summary_rows


def main():
    perturbation_args = parse_perturbation_arguments()
    args = get_args(
        model_name='CMFC',
        dataset_class='Planetoid',
        dataset_name=None,
        custom_key='classification',
    )
    args.p = float(
        perturbation_args.p
        if perturbation_args.p is not None
        else getattr(args, 'p', 0.0)
    )
    args.feature_mask_ratio = float(
        perturbation_args.feature_mask_ratio
        if perturbation_args.feature_mask_ratio is not None
        else getattr(args, 'feature_mask_ratio', 0.0)
    )
    validate_controlled_perturbation(args)

    pprint_args(args)
    running_device = resolve_configured_device(args)
    print(f'Using device: {running_device}')

    if args.sweep_enabled:
        run_perturbation_sweep(args, running_device)
        return

    (
        mean_accuracy,
        confidence_interval,
        mean_edge_retention,
        edge_retention_interval,
        mean_feature_retention,
        feature_retention_interval,
    ) = run_repeated_perturbation_experiment(args, running_device)
    print(
        f"\n{'=' * 30}\n"
        f'Perturbation ablation results over {args.num_runs} runs:\n'
        f'p: {args.p:.4f}\n'
        f'feature_mask_ratio: {args.feature_mask_ratio:.4f}\n'
        f'Actual edge retention: '
        f'{mean_edge_retention:.6f} '
        f'+/- {edge_retention_interval:.6f}\n'
        f'Actual attribute retention: '
        f'{mean_feature_retention:.6f} '
        f'+/- {feature_retention_interval:.6f}\n'
        f'Acc: {mean_accuracy:.2f}% +/- {confidence_interval:.2f}%\n'
        f"{'=' * 30}"
    )


if __name__ == '__main__':
    main()
