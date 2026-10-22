# CMFC

## Installation

```bash
conda create -n cmfc python=3.10
conda activate cmfc
python -m pip install torch torch-geometric numpy scikit-learn ruamel.yaml tqdm termcolor
```

For GPU support, install the appropriate [PyTorch](https://pytorch.org/get-started/locally/) version for your CUDA environment.

## Running

Run the following command from the project root:

```bash
python train.py --dataset-name Cora --n 2 --k 1 --q 20 --num_runs 20 --device auto
```

This runs a 2-way 1-shot experiment on Cora with 20 query samples per class across 20 independent runs. Other parameters are loaded from `args.yaml`. The dataset is downloaded automatically on the first run. With `--device auto`, the program uses a GPU when CUDA is available and otherwise uses the CPU.
