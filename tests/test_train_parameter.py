import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import train_parameter


def make_args(**overrides):
    values = {
        "sweep_enabled": True,
        "sweep_parameter": "m",
        "sweep_values": [3, 5],
        "sweep_output_dir": "results",
        "dataset_name": "Cora",
        "n": 2,
        "k": 1,
        "m": 1,
        "num_runs": 2,
        "seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainParameterTests(unittest.TestCase):
    def test_result_path_matches_required_layout(self):
        args = make_args()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = train_parameter.parameter_result_path(
                args,
                0.1,
                ablation_root=temp_dir,
            )

        self.assertTrue(path.endswith(os.path.join(
            "m",
            "Cora",
            "2-way 1-shot para-0.1.csv",
        )))

    def test_disabled_sweep_does_not_run_or_create_output(self):
        args = make_args(sweep_enabled=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = os.path.join(temp_dir, "ablation")
            with patch.object(
                train_parameter.base,
                "run_repeated_experiment",
            ) as run:
                results = train_parameter.run_parameter_sweep(
                    args,
                    None,
                    ablation_root=output,
                )

            self.assertEqual(results, [])
            run.assert_not_called()
            self.assertFalse(os.path.exists(output))

    def test_each_sweep_value_gets_a_detailed_csv(self):
        args = make_args(sweep_values=[3, 5])

        def fake_repeated(run_args, _device, graph_cache, result_path):
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(
                result_path,
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "record_type",
                    "run",
                    "seed",
                    "accuracy_percent",
                    "mean_accuracy_percent",
                    "confidence_interval_95_percent",
                ])
                writer.writeheader()
                writer.writerow({
                    "record_type": "run",
                    "run": 1,
                    "seed": run_args.seed,
                    "accuracy_percent": 80 + run_args.m,
                })
                writer.writerow({
                    "record_type": "summary",
                    "mean_accuracy_percent": 80 + run_args.m,
                    "confidence_interval_95_percent": 1.5,
                })
            return 80 + run_args.m, 1.5

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                train_parameter.base,
                "run_repeated_experiment",
                side_effect=fake_repeated,
            ) as run:
                results = train_parameter.run_parameter_sweep(
                    args,
                    None,
                    ablation_root=temp_dir,
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual([item["value"] for item in results], [3, 5])
            self.assertEqual(
                [os.path.basename(item["csv_path"]) for item in results],
                [
                    "2-way 1-shot para-3.csv",
                    "2-way 1-shot para-5.csv",
                ],
            )
            for item in results:
                self.assertTrue(os.path.isfile(item["csv_path"]))
                with open(
                    item["csv_path"],
                    encoding="utf-8-sig",
                    newline="",
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(
                    [row["record_type"] for row in rows],
                    ["run", "summary"],
                )

    def test_single_sweep_value_runs_one_parameter_setting(self):
        args = make_args(sweep_values=["4"])
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                train_parameter.base,
                "run_repeated_experiment",
                return_value=(75.0, 0.0),
            ) as run:
                results = train_parameter.run_parameter_sweep(
                    args,
                    None,
                    ablation_root=temp_dir,
                )

        run.assert_called_once()
        self.assertEqual(results[0]["value"], 4)


if __name__ == "__main__":
    unittest.main()
