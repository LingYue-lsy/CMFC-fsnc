import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train
import train_label


class TinyCMFC(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features, _adjacency):
        embeddings = features * self.scale
        return embeddings, embeddings, embeddings


class TrainLabelTests(unittest.TestCase):
    def test_label_match_metrics_cover_full_partial_and_zero_matches(self):
        labels = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([0, 2])

        cases = [
            (torch.tensor([[0, 1], [2, 3]]), (4, 4, 1.0)),
            (torch.tensor([[0, 2], [1, 3]]), (2, 4, 0.5)),
            (torch.tensor([[2, 3], [0, 1]]), (0, 4, 0.0)),
        ]
        for selected, expected in cases:
            with self.subTest(expected=expected):
                actual = train_label.label_match_metrics(
                    labels,
                    targets,
                    selected,
                )
                self.assertEqual(actual, expected)
                self.assertGreaterEqual(actual[2], 0.0)
                self.assertLessEqual(actual[2], 1.0)

    def test_label_csv_path_contains_setting_and_seed(self):
        args = SimpleNamespace(
            dataset_name="Cora",
            n=2,
            k=5,
            seed=1001,
            support_selection="cosine",
        )
        with tempfile.TemporaryDirectory() as output_root:
            path = train_label.label_csv_path(args, output_root)
        self.assertTrue(path.endswith(os.path.join(
            "Cora",
            "cosine",
            "2-way 5-shot seed-1001.csv",
        )))

    def test_training_writes_one_row_per_actual_epoch_and_evaluates(self):
        labels = torch.tensor([0, 0, 1, 1])
        features = torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        adjacency = torch.eye(4).to_sparse()
        prepared = train.PreparedGraph(
            dataset_name="Tiny",
            features=features,
            adjacency=adjacency,
            labels=labels,
            labels_cpu=labels,
            neighbor_table=train.build_neighbor_table(adjacency),
            num_features=2,
            num_nodes=4,
            default_masks=None,
        )
        mask = torch.ones(4, dtype=torch.bool)
        masks = train.DatasetMasks(mask, mask, mask)
        args = SimpleNamespace(
            dataset_name="Tiny",
            seed=7,
            n=2,
            k=1,
            m=1,
            task_num=1,
            NN=1,
            margin1=0.4,
            margin2=0.2,
            w_loss1=1.0,
            w_loss2=1.0,
            w_loss3=1.0,
            cfg=[2],
            dropout=0.0,
            lr=0.01,
            weight_decay=0.0,
            epochs=4,
            start_epoch=1,
            support_selection="cosine",
            save_model=False,
            checkpoint_dir="checkpoints",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "label.csv")
            with patch.object(train_label, "CMFC_Fast", TinyCMFC):
                with patch.object(
                    train,
                    "evaluate_on_test_set",
                    return_value=75.0,
                ) as evaluate:
                    accuracy, returned_path = train_label.run_label_training(
                        args,
                        torch.device("cpu"),
                        prepared_graph=prepared,
                        masks=masks,
                        csv_path=csv_path,
                    )

            with open(csv_path, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = reader.fieldnames

        self.assertEqual(columns, train_label.LABEL_COLUMNS)
        self.assertEqual(len(rows), args.epochs - args.start_epoch)
        self.assertEqual([row["epoch"] for row in rows], ["2", "3", "4"])
        self.assertEqual(returned_path, csv_path)
        self.assertEqual(accuracy, 75.0)
        evaluate.assert_called_once()
        for row in rows:
            probability = float(row["label_match_probability"])
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
