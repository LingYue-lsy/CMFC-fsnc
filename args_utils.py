import os
import argparse
from ruamel.yaml import YAML

def get_args_key(args):
    """
    生成用于在YAML配置中查找参数的键
    """
    return "-".join([args.model_name, args.dataset_name, args.custom_key])

def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    """
    解析命令行参数并从YAML配置文件中加载特定于实验的参数
    
    Args:
        model_name: 模型名称
        dataset_class: 数据集类别
        dataset_name: 数据集名称
        custom_key: 自定义键，用于在YAML中查找特定配置
        yaml_path: YAML配置文件路径
        
    Returns:
        argparse.Namespace: 解析后的参数
    """
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    with open(yaml_path, encoding="utf-8") as args_file:
        yaml_config = YAML().load(args_file)

    model_defaults = yaml_config.get(model_name, {})
    dataset_name = model_defaults.get("dataset_name", dataset_name)

    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom_key", default=custom_key)
    parser.add_argument("--save_model", default=False)
    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--gpu-id', '--gpu_id', dest='gpu_id', type=int, default=0)
    # Pretrain
    parser.add_argument("--pretrain", default=False, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    parser.add_argument("--seed", default=1)
    parser.add_argument('--n', type=int, default=2, help='')
    parser.add_argument('--k', type=int, default=1, help='')
    parser.add_argument('--q', type=int, default=20, help='')
    parser.add_argument('--task_num', type=int, default=50, help='')
    parser.add_argument('--num_runs', type=int, default=20, help='')
    parser.add_argument('--m', type=int, default=5,
                        help='number of sampled neighbors to aggregate')
    parser.add_argument(
        '--sweep-enabled',
        '--sweep_enabled',
        dest='sweep_enabled',
        action='store_true',
        default=False,
        help='enable a single-parameter sweep',
    )
    parser.add_argument('--sweep-parameter', '--sweep_parameter',
                        dest='sweep_parameter', default='m')
    parser.add_argument('--sweep-values', '--sweep_values',
                        dest='sweep_values', nargs='+', default=[])
    parser.add_argument('--sweep-output-dir', '--sweep_output_dir',
                        dest='sweep_output_dir', default='results')
    parser.add_argument(
        '--support-selection',
        '--support_selection',
        dest='support_selection',
        default='cosine',
        choices=['cosine', 'euclidean', 'random'],
        help='strategy used to choose top-k pseudo support nodes',
    )

    parser.add_argument('--use_val_set', type=bool, default=False, help='')
    parser.add_argument('--early_stopping', type=bool, default=False, help='')
    parser.add_argument('--patience', type=int, default=20, help='')
    parser.add_argument('--eval_interval', type=int, default=10, help='')

    

    
    # Experiment specific parameters loaded from .yamls
    args = parser.parse_args()
    args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
    try:
        parser.set_defaults(**dict(yaml_config[args_key].items()))
    except KeyError:
        raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    
    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(args):
    """
    打印参数信息
    """
    print("==========       CONFIG      =============")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("==========     CONFIG END    =============")
    print("\n")

def print_args_as_table(args):
    """
    以表格形式打印参数
    """
    header = "|Parameter|Value|"
    line = "|---|---|"
    rows = []
    for arg, content in sorted(args.__dict__.items()):
        row = f"|{arg}|{content}|"
        rows.append(row)
    
    print(header)
    print(line)
    for row in rows:
        print(row)
    print("\n")
