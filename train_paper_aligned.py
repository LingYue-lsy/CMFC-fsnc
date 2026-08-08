"""Training entry point for the manuscript-aligned CMFC encoder.

This file intentionally reuses the data preparation, task construction,
losses, evaluation, and configuration handling from train.py. Only the model
class is replaced, so the original training and model files remain unchanged.
"""

import copy
import os.path as osp

import train as original_train

from models.model_CMFC_paper_aligned import CMFC_PaperAligned


# Functions defined in train.py resolve CMFC_Fast from that module's globals.
# Rebinding it here keeps the complete original training pipeline while using
# the manuscript-aligned encoder.
original_train.CMFC_Fast = CMFC_PaperAligned


def paper_aligned_result_csv_path(args):
    """Keep paper-aligned results separate from results of the original model."""
    result_dir = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "res_paper_aligned",
        str(args.dataset_name),
    )
    return osp.join(result_dir, f"{args.n}-way {args.k}-shot.csv")


if __name__ == "__main__":
    main_args = original_train.get_args(
        model_name="CMFC",
        dataset_class="Planetoid",
        dataset_name=None,
        custom_key="classification",
    )

    original_train.pprint_args(main_args)
    running_device = original_train.resolve_configured_device(main_args)
    print(f"Using device: {running_device}")
    print("Encoder variant: manuscript-aligned layer-wise EMA-GCN")

    if main_args.sweep_enabled:
        sweep_args = copy.deepcopy(main_args)
        sweep_args.sweep_output_dir = osp.join(
            sweep_args.sweep_output_dir,
            "paper_aligned",
        )
        original_train.run_parameter_sweep(sweep_args, running_device)
    else:
        mean_accuracy, confidence_interval = (
            original_train.run_repeated_experiment(
                main_args,
                running_device,
                result_path=paper_aligned_result_csv_path(main_args),
            )
        )
        print(
            f"\n{'=' * 30}\n"
            f"Final Results over {main_args.num_runs} runs:\n"
            f"Acc: {mean_accuracy:.2f}% +/- {confidence_interval:.2f}%\n"
            f"{'=' * 30}"
        )
