import contextlib
import csv
import io
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

import train


class StaticEmbeddingModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def eval(self):
        return self

    def embed(self, features, adjacency):
        return self.embeddings, self.embeddings, self.embeddings


def legacy_dense_adjacency(edge_index, num_nodes):
    edge_index = to_undirected(edge_index, num_nodes)
    weights = torch.ones(edge_index.size(1))
    adjacency = torch.sparse_coo_tensor(
        edge_index,
        weights,
        (num_nodes, num_nodes),
    ).to_dense()
    adjacency = adjacency + torch.eye(num_nodes)
    eps = 2.2204e-16
    degree_inv_sqrt = (
        adjacency.sum(dim=-1).clamp(min=0) + eps
    ).pow(-0.5)
    return (
        degree_inv_sqrt.unsqueeze(-1)
        * adjacency
        * degree_inv_sqrt.unsqueeze(-2)
    )


def legacy_neighbor_table(adjacency, sample_count=100):
    row, col = adjacency.indices()
    degrees = torch.bincount(row, minlength=adjacency.size(0)).tolist()
    offsets = [0]
    for degree in degrees:
        offsets.append(offsets[-1] + degree)
    rows = []
    for sample in range(1, sample_count + 1):
        positions = [
            offsets[node] + sample % degrees[node]
            for node in range(adjacency.size(0))
        ]
        rows.append(col[positions])
    return torch.stack(rows)


def legacy_margin_terms(support, neighbor, negative, margin, mask_margin):
    margin_loss = torch.nn.MarginRankingLoss(
        margin=margin,
        reduction='none',
    )
    zero = torch.tensor([0.0])
    support_term = 0
    neighbor_term = 0
    mask_term = 0
    for negative_row in negative:
        for support_idx in range(support.size(1)):
            support_row = support[:, support_idx]
            neighbor_row = neighbor[:, support_idx]
            label = -torch.ones_like(support_row)
            support_term += margin_loss(
                support_row,
                negative_row,
                label,
            ).mean()
            neighbor_term += margin_loss(
                neighbor_row,
                negative_row,
                label,
            ).mean()
            mask_term += torch.max(
                negative_row - support_row.detach() - mask_margin,
                zero,
            ).sum()
    return support_term, neighbor_term, mask_term


def legacy_evaluate(model, prepared_graph, test_mask, args):
    with torch.no_grad():
        _, embeddings, _ = model.embed(
            prepared_graph.features,
            prepared_graph.adjacency,
        )
        embeddings = embeddings / embeddings.norm(dim=1)[:, None]

    scores = []
    for _ in range(100):
        eval_indices = torch.nonzero(test_mask).squeeze().cpu().numpy()
        eval_labels = prepared_graph.labels[eval_indices].cpu().numpy()
        available_classes = np.unique(eval_labels)
        n_way = min(args.n, len(available_classes))
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
            class_indices = eval_indices[eval_labels == class_idx]
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
        classifier = train.LogisticRegression(solver='lbfgs', max_iter=1000)
        classifier.fit(
            embeddings[support_idx].cpu().numpy(),
            np.array(support_labels),
        )
        scores.append(
            classifier.score(
                embeddings[query_idx].cpu().numpy(),
                np.array(query_labels),
            )
        )
    return np.mean(scores) * 100


class TrainOptimizationTests(unittest.TestCase):
    def test_epoch_range_uses_epochs_as_total_update_count(self):
        args = SimpleNamespace(epochs=1000, start_epoch=0)
        epochs = train.training_epoch_range(args)
        self.assertEqual(len(epochs), 1000)
        self.assertEqual(epochs.start, 0)
        self.assertEqual(epochs.stop, 1000)

        resumed = train.training_epoch_range(
            SimpleNamespace(epochs=1000, start_epoch=400)
        )
        self.assertEqual(len(resumed), 600)
        self.assertEqual(resumed.start, 400)
        self.assertEqual(resumed.stop, 1000)

    def test_epoch_range_rejects_invalid_boundaries(self):
        invalid_values = [(0, 0), (10, -1), (10, 10), (10, 11)]
        for epochs, start_epoch in invalid_values:
            with self.subTest(epochs=epochs, start_epoch=start_epoch):
                with self.assertRaises(ValueError):
                    train.training_epoch_range(
                        SimpleNamespace(
                            epochs=epochs,
                            start_epoch=start_epoch,
                        )
                    )

    def test_configured_device_selection(self):
        cpu_args = SimpleNamespace(device='cpu', gpu_id=3)
        self.assertEqual(
            train.resolve_configured_device(cpu_args),
            torch.device('cpu'),
        )

        auto_args = SimpleNamespace(device='auto', gpu_id=0)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(
                train.resolve_configured_device(auto_args),
                torch.device('cpu'),
            )

        cuda_args = SimpleNamespace(device='cuda', gpu_id=0)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                train.resolve_configured_device(cuda_args)

        selected_args = SimpleNamespace(device='auto', gpu_id=1)
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'device_count', return_value=2):
                self.assertEqual(
                    train.resolve_configured_device(selected_args),
                    torch.device('cuda:1'),
                )

        invalid_args = SimpleNamespace(device='cuda', gpu_id=2)
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'device_count', return_value=2):
                with self.assertRaises(ValueError):
                    train.resolve_configured_device(invalid_args)

    def test_sparse_adjacency_matches_dense_formula(self):
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 2],
            [1, 0, 0, 2, 3, 3],
        ])
        expected = legacy_dense_adjacency(edge_index, num_nodes=5)
        actual = train.build_normalized_sparse_adjacency(
            edge_index,
            num_nodes=5,
        ).to_dense()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_neighbor_table_matches_legacy_sampling(self):
        edge_index = torch.tensor([
            [0, 1, 1, 2, 3],
            [1, 0, 2, 1, 3],
        ])
        adjacency = train.build_normalized_sparse_adjacency(edge_index, 4)
        expected = legacy_neighbor_table(adjacency)
        actual = train.build_neighbor_table(adjacency)
        self.assertTrue(torch.equal(actual, expected))

    def test_batched_distances_and_losses_match_legacy_loops(self):
        torch.manual_seed(7)
        node_embeddings = torch.randn(7, 5)
        propagated = torch.randn(7, 5)
        task_embeddings = node_embeddings[:4]
        topk_indices = torch.tensor([
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
        ])
        negative_indices = torch.tensor([
            [1, 2, 3, 0],
            [2, 3, 0, 1],
        ])
        neighbor_embeddings = torch.randn(4, 2, 5)

        support, neighbor, negative = train.compute_batched_distances(
            task_embeddings,
            propagated,
            neighbor_embeddings,
            topk_indices,
            negative_indices,
        )

        expected_support = []
        expected_neighbor = []
        for task_idx in range(task_embeddings.size(0)):
            anchor = task_embeddings[task_idx].unsqueeze(0)
            support_nodes = propagated[topk_indices[task_idx]]
            expected_support.append(F.pairwise_distance(
                anchor.expand_as(support_nodes),
                support_nodes,
            ))
            expected_neighbor.append(F.pairwise_distance(
                anchor.expand_as(neighbor_embeddings[task_idx]),
                neighbor_embeddings[task_idx],
            ))
        expected_negative = torch.stack([
            F.pairwise_distance(
                task_embeddings,
                task_embeddings[index_row],
            )
            for index_row in negative_indices
        ])
        self.assertTrue(torch.allclose(support, torch.stack(expected_support)))
        self.assertTrue(torch.allclose(neighbor, torch.stack(expected_neighbor)))
        self.assertTrue(torch.allclose(negative, expected_negative))

        expected_terms = legacy_margin_terms(
            support,
            neighbor,
            negative,
            margin=0.4,
            mask_margin=1.0,
        )
        actual_terms = train.compute_batched_margin_terms(
            support,
            neighbor,
            negative,
            margin=0.4,
            mask_margin=1.0,
        )
        for actual, expected in zip(actual_terms, expected_terms):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_seeded_class_masks_match_legacy_membership(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        for seed in (1, 1001, 2001):
            args = SimpleNamespace(seed=seed)
            torch.manual_seed(seed)
            classes = torch.arange(5)[torch.randperm(5)]
            expected = train.DatasetMasks(
                torch.isin(labels, classes[:2]),
                torch.isin(labels, classes[2:3]),
                torch.isin(labels, classes[3:]),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                actual = train.make_class_split_masks(labels, args, 2, 1)
            self.assertTrue(torch.equal(actual.train, expected.train))
            self.assertTrue(torch.equal(actual.val, expected.val))
            self.assertTrue(torch.equal(actual.test, expected.test))

    def test_graph_cache_is_shared_across_repeated_experiments(self):
        args = SimpleNamespace(
            num_runs=3,
            seed=1,
            dataset_name='Cora',
        )
        cache = {}
        prepared = object()
        with patch.object(train, 'prepare_graph', return_value=prepared) as load:
            with patch.object(train, 'run_CMFC', return_value=50.0) as run:
                train.run_repeated_experiment(args, None, cache)
                train.run_repeated_experiment(args, None, cache)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(run.call_count, 6)

    def test_repeated_experiment_records_runs_splits_and_summary(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        prepared = train.PreparedGraph(
            dataset_name='Cora',
            features=torch.empty(10, 1),
            adjacency=torch.eye(10).to_sparse(),
            labels=labels,
            labels_cpu=labels,
            neighbor_table=torch.empty(0, dtype=torch.long),
            num_features=1,
            num_nodes=10,
            default_masks=None,
        )
        args = SimpleNamespace(
            num_runs=2,
            seed=1,
            dataset_name='Cora',
            n=2,
            k=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'result.csv')
            with patch.object(
                train,
                'get_cached_prepared_graph',
                return_value=prepared,
            ):
                with patch.object(
                    train,
                    'run_CMFC',
                    side_effect=[50.0, 70.0],
                ):
                    mean, interval = train.run_repeated_experiment(
                        args,
                        None,
                        result_path=csv_path,
                    )

            with open(csv_path, encoding='utf-8-sig', newline='') as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual([row['record_type'] for row in rows], [
            'run', 'run', 'summary',
        ])
        self.assertEqual([row['seed'] for row in rows[:2]], ['1', '1001'])
        self.assertIn('"type":"classes"', rows[0]['train_split'])
        self.assertEqual(rows[0]['accuracy_percent'], '50.000000')
        self.assertEqual(float(rows[-1]['mean_accuracy_percent']), mean)
        self.assertEqual(
            float(rows[-1]['confidence_interval_95_percent']),
            round(interval, 6),
        )
        self.assertEqual(mean, 60.0)

    def test_evaluation_preprocessing_preserves_result(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        embeddings = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [1.1, 0.1], [0.8, 0.2],
            [0.0, 1.0], [0.1, 0.9], [0.1, 1.1], [0.2, 0.8],
        ])
        identity = torch.eye(8).to_sparse()
        prepared = train.PreparedGraph(
            dataset_name='test',
            features=embeddings,
            adjacency=identity,
            labels=labels,
            labels_cpu=labels,
            neighbor_table=torch.empty(0, dtype=torch.long),
            num_features=2,
            num_nodes=8,
            default_masks=None,
        )
        model = StaticEmbeddingModel(embeddings)
        args = SimpleNamespace(n=2, k=1, q=1)
        test_mask = torch.ones(8, dtype=torch.bool)

        np.random.seed(9)
        expected = legacy_evaluate(model, prepared, test_mask, args)
        np.random.seed(9)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = train.evaluate_on_test_set(
                model,
                prepared,
                test_mask,
                args,
            )
        self.assertTrue(math.isclose(actual, expected, abs_tol=1e-12))


if __name__ == '__main__':
    unittest.main()
