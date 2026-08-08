import unittest

import torch

from models import CMFC_Fast, LogReg


class ModelCleanupTests(unittest.TestCase):
    def test_state_dict_keys_and_initial_ema_copy_are_preserved(self):
        torch.manual_seed(4)
        model = CMFC_Fast(4, [8, 4])
        self.assertEqual(
            list(model.state_dict()),
            [
                'MLP.0.weight',
                'MLP.0.bias',
                'MLP.2.weight',
                'MLP.2.bias',
                'MLP_ema.0.weight',
                'MLP_ema.0.bias',
                'MLP_ema.2.weight',
                'MLP_ema.2.bias',
            ],
        )
        for main_param, ema_param in zip(
            model.MLP.parameters(),
            model.MLP_ema.parameters(),
        ):
            self.assertTrue(torch.equal(main_param, ema_param))
            self.assertFalse(ema_param.requires_grad)

    def test_ema_update_matches_original_formula(self):
        torch.manual_seed(5)
        model = CMFC_Fast(3, [6, 3], momentum=0.9)
        previous_ema = [
            parameter.clone() for parameter in model.MLP_ema.parameters()
        ]
        with torch.no_grad():
            for parameter in model.MLP.parameters():
                parameter.add_(0.25)
        expected = [
            old * 0.9 + current * 0.1
            for old, current in zip(previous_ema, model.MLP.parameters())
        ]

        model._update_ema_weights()

        for actual, expected_parameter in zip(
            model.MLP_ema.parameters(),
            expected,
        ):
            self.assertTrue(torch.allclose(actual, expected_parameter))

    def test_dense_and_sparse_aggregation_match(self):
        torch.manual_seed(6)
        model = CMFC_Fast(4, [8, 4], dropout=0.3)
        model.eval()
        features = torch.randn(5, 4)
        dense_adjacency = torch.tensor([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ])

        dense_outputs = model(features, dense_adjacency)
        sparse_outputs = model(features, dense_adjacency.to_sparse())

        for dense, sparse in zip(dense_outputs, sparse_outputs):
            self.assertTrue(torch.allclose(dense, sparse, atol=1e-7))

    def test_redundant_runtime_state_is_removed(self):
        model = CMFC_Fast(2, [2])
        self.assertFalse(hasattr(model, 'act'))
        self.assertFalse(hasattr(model, 'A'))
        self.assertFalse(hasattr(model, 'sparse'))

    def test_logreg_initialization_remains_valid(self):
        model = LogReg(3, 2)
        self.assertTrue(torch.equal(model.fc.bias, torch.zeros_like(model.fc.bias)))
        self.assertEqual(tuple(model(torch.ones(4, 3)).shape), (4, 2))


if __name__ == '__main__':
    unittest.main()
