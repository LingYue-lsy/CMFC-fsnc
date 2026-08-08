"""Run CMFC parameter sweeps and save one detailed CSV per value.

Results are written to::

    results/ablation/<parameter>/<dataset>/
        <n>-way <k>-shot para-<value>.csv

Each CSV contains one row per seeded run followed by a summary row with the
mean accuracy and its 95% confidence interval.  This entry point intentionally
does nothing unless ``sweep_enabled`` is true.
"""

from __future__ import annotations

import copy
import os
import os.path as osp
import re

import train as base
from args_utils import get_args, pprint_args


PROJECT_ROOT = osp.dirname(osp.realpath(__file__))
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_path_component(value, field_name):
    """Return a Windows-safe path component without changing normal values."""
    component = str(value).strip()
    if not component or component in {".", ".."}:
        raise ValueError(f"{field_name} must not be empty or relative traversal.")
    if INVALID_PATH_CHARS.search(component):
        raise ValueError(
            f"{field_name} contains a character that is invalid in a path: "
            f"{component!r}."
        )
    return component


def format_parameter_value(value):
    """Format a sweep value consistently for console output and file names."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def parameter_result_path(args, parameter_value, ablation_root=None):
    """Return the detailed CSV path for one parameter value."""
    parameter_name = _safe_path_component(
        args.sweep_parameter,
        "sweep_parameter",
    )
    dataset_name = _safe_path_component(args.dataset_name, "dataset_name")
    value_text = _safe_path_component(
        format_parameter_value(parameter_value),
        "sweep value",
    )

    if ablation_root is None:
        configured_root = osp.expanduser(
            str(getattr(args, "sweep_output_dir", "results"))
        )
        if not osp.isabs(configured_root):
            configured_root = osp.join(PROJECT_ROOT, configured_root)
        ablation_root = osp.join(configured_root, "ablation")

    filename = (
        f"{int(args.n)}-way {int(args.k)}-shot para-{value_text}.csv"
    )
    return osp.abspath(
        osp.join(ablation_root, parameter_name, dataset_name, filename)
    )


def _validated_sweep_values(args):
    parameter_name = str(args.sweep_parameter).strip()
    if not parameter_name or parameter_name.startswith("sweep_"):
        raise ValueError(f"Invalid sweep parameter: {parameter_name!r}.")
    if not hasattr(args, parameter_name):
        raise ValueError(f"Unknown sweep parameter: {parameter_name!r}.")
    if not args.sweep_values:
        raise ValueError("sweep_values must contain at least one value.")

    current_value = getattr(args, parameter_name)
    return [
        base.coerce_sweep_value(value, current_value)
        for value in args.sweep_values
    ]


def run_parameter_sweep(args, running_device, ablation_root=None):
    """Run every configured sweep value and create one detailed CSV each."""
    if not bool(args.sweep_enabled):
        print("Parameter sweep is disabled; no experiment was run.")
        return []

    parameter_name = str(args.sweep_parameter).strip()
    sweep_values = _validated_sweep_values(args)
    graph_cache = {}
    results = []

    for value in sweep_values:
        sweep_args = copy.deepcopy(args)
        setattr(sweep_args, parameter_name, value)
        csv_path = parameter_result_path(
            sweep_args,
            value,
            ablation_root=ablation_root,
        )
        os.makedirs(osp.dirname(csv_path), exist_ok=True)

        print(
            f"\n{'=' * 30}\n"
            f"Sweep: {parameter_name}={format_parameter_value(value)}\n"
            f"Output: {csv_path}\n"
            f"{'=' * 30}"
        )
        mean_accuracy, confidence_interval = base.run_repeated_experiment(
            sweep_args,
            running_device,
            graph_cache=graph_cache,
            result_path=csv_path,
        )
        results.append({
            "parameter": parameter_name,
            "value": value,
            "csv_path": csv_path,
            "mean_accuracy_percent": mean_accuracy,
            "confidence_interval_95_percent": confidence_interval,
        })
        print(
            f"{parameter_name}={format_parameter_value(value)}: "
            f"{mean_accuracy:.2f}% +/- {confidence_interval:.2f}%"
        )

    print(f"\nCompleted {len(results)} parameter setting(s).")
    return results


def main():
    args = get_args(
        model_name="CMFC",
        dataset_class="Planetoid",
        dataset_name=None,
        custom_key="classification",
    )
    pprint_args(args)

    if not args.sweep_enabled:
        print("Parameter sweep is disabled; set sweep_enabled: True to run.")
        return

    running_device = base.resolve_configured_device(args)
    print(f"Using device: {running_device}")
    run_parameter_sweep(args, running_device)


if __name__ == "__main__":
    main()
