from time import perf_counter as t
import copy
import csv
from dataclasses import dataclass
import json
import random
import argparse
from pathlib import Path
from ruamel.yaml import YAML
import torch.nn as nn
import torch
from torch_geometric.utils import coalesce, to_undirected
import os.path as osp
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import Amazon, CitationFull, Coauthor, Planetoid
import os
from sklearn.linear_model import LogisticRegression
from termcolor import cprint


def get_args(argv=None):
    """Load dataset defaults and apply explicit command-line overrides."""
    base = Path(__file__).resolve().parent
    with (base / "args.yaml").open(encoding="utf-8") as stream:
        config = YAML(typ="safe").load(stream)
    datasets = ("Cora", "CiteSeer", "CoraFull", "CS", "Computers")
    parser = argparse.ArgumentParser(description="CMFC self-supervised few-shot node classification")
    parser.add_argument("--dataset-name", choices=datasets, default=config["CMFC"]["dataset_name"])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=0)
    parser.add_argument("--data-root", default=str(base / "dataset"))
    parser.add_argument("--output-dir", default=str(base / "res"))
    parser.add_argument("--checkpoint_dir", default=str(base / "checkpoints"))
    parser.add_argument("--save-model", "--save_model", dest="save_model", action="store_true")
    integer_defaults = {"n": 2, "k": 1, "q": 20, "seed": 1, "num_runs": 5,
                        "epochs": 1000, "task_num": 50, "m": 5, "NN": 4}
    float_defaults = {"lr": 0.005, "margin1": 0.8, "margin2": 0.2,
                      "w_loss1": 5.0, "w_loss2": 5.0, "w_loss3": 1.0,
                      "weight_decay": 0.00005, "dropout": 0.3}
    for name, default in integer_defaults.items():
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in float_defaults.items():
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--cfg", type=int, nargs="+", default=[512, 128])
    selected, _ = parser.parse_known_args(argv)
    defaults = dict(config["CMFC"])
    defaults.update(config["CMFC-" + selected.dataset_name + "-classification"])
    allowed = {action.dest for action in parser._actions}
    unknown = set(defaults) - allowed
    if unknown:
        parser.error("Unsupported configuration keys: " + ", ".join(sorted(unknown)))
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in integer_defaults:
        if name != "seed" and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    return args


def pprint_args(args):
    print("========== CONFIG ==========")
    for name, value in sorted(vars(args).items()):
        print(f"{name}: {value}")
    print("============================")


def make_mlp(in_channels, cfg, dropout):
    """Build the online and exponential-moving-average MLPs."""
    if not cfg:
        raise ValueError("cfg must contain at least the output dimension.")

    layers = []
    current_channels = in_channels
    final_index = len(cfg) - 1

    for index, out_channels in enumerate(cfg):
        layers.append(nn.Linear(current_channels, out_channels))
        if index != final_index:
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        current_channels = out_channels

    return nn.Sequential(*layers)


class CMFC_Fast(nn.Module):
    """CMFC encoder with an EMA MLP and layer-wise graph propagation.

    The online and EMA MLPs have identical architectures. The structural
    branch reuses the EMA projection weights and performs normalized graph
    propagation at every layer; it has no independent trainable parameters.
    """

    def __init__(self, n_in, cfg=None, dropout=0.2, momentum=0.99):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided.")

        self.cfg = tuple(cfg)
        self.dropout = dropout
        self.momentum = momentum
        self.MLP = make_mlp(n_in, self.cfg, dropout)
        self.MLP_ema = make_mlp(n_in, self.cfg, dropout)

        for module in self.modules():
            self.weights_init(module)

        self._copy_weights()
        self.MLP_ema.requires_grad_(False)

    @torch.no_grad()
    def _copy_weights(self):
        for online_parameter, ema_parameter in zip(
            self.MLP.parameters(),
            self.MLP_ema.parameters(),
        ):
            ema_parameter.copy_(online_parameter)

    @torch.no_grad()
    def _update_ema_weights(self):
        for online_parameter, ema_parameter in zip(
            self.MLP.parameters(),
            self.MLP_ema.parameters(),
        ):
            ema_parameter.mul_(self.momentum)
            ema_parameter.add_(
                online_parameter,
                alpha=1.0 - self.momentum,
            )

    @staticmethod
    def weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _aggregate(adj, embeddings):
        return torch.sparse.mm(adj, embeddings)

    def _ema_gcn(self, features, adj):
        """Apply normalized graph propagation layer by layer."""
        h = features
        for layer in self.MLP_ema:
            if isinstance(layer, nn.Linear):
                # Reuse the EMA projection matrix without an extra branch.
                h = F.linear(h, layer.weight, bias=None)
                h = self._aggregate(adj, h)
                h = F.relu(h)
        return h

    def forward(self, seq_a, adj=None):
        if adj is None:
            raise ValueError("adj must be provided.")

        # Dropout is applied only after hidden activations.
        h_a = self.MLP(seq_a)

        if self.training:
            self._update_ema_weights()

        with torch.no_grad():
            h_a_ema = self.MLP_ema(seq_a)
            h_p = self._ema_gcn(seq_a, adj)

        return h_a, h_p, h_a_ema

    def embed(self, seq_a, adj=None):
        if adj is None:
            raise ValueError("adj must be provided.")

        with torch.no_grad():
            h_a_ema = self.MLP_ema(seq_a)
            h_p = self._ema_gcn(seq_a, adj)

        return h_a_ema, h_p, h_a_ema


@dataclass
class DatasetMasks:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


@dataclass
class PreparedGraph:
    dataset_name: str
    features: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor
    labels_cpu: torch.Tensor
    neighbor_table: torch.Tensor
    num_features: int
    num_nodes: int


def resolve_device(gpu_id=None):
    if isinstance(gpu_id, torch.device):
        return gpu_id
    if gpu_id is None:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def resolve_configured_device(args):
    if args.device == 'cpu':
        return torch.device('cpu')

    if not torch.cuda.is_available():
        if args.device == 'cuda':
            raise RuntimeError('device=cuda was requested, but CUDA is unavailable.')
        return torch.device('cpu')

    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"gpu_id={args.gpu_id} is outside the available range "
            f"[0, {torch.cuda.device_count() - 1}]."
        )
    return torch.device(f'cuda:{args.gpu_id}')


def build_normalized_sparse_adjacency(edge_index, num_nodes):
    edge_index = to_undirected(edge_index, num_nodes)
    self_loops = torch.arange(num_nodes, device=edge_index.device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    edge_weight = torch.ones(
        edge_index.size(1),
        dtype=torch.float32,
        device=edge_index.device,
    )
    edge_index, edge_weight = coalesce(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
        reduce='sum',
    )

    row, col = edge_index
    weighted_degree = torch.zeros(
        num_nodes,
        dtype=edge_weight.dtype,
        device=edge_weight.device,
    )
    weighted_degree.index_add_(0, row, edge_weight)
    degree_inv_sqrt = weighted_degree.pow(-0.5)
    normalized_weight = (
        degree_inv_sqrt[row] * edge_weight * degree_inv_sqrt[col]
    )
    return torch.sparse_coo_tensor(
        edge_index,
        normalized_weight,
        (num_nodes, num_nodes),
    ).coalesce()


def build_neighbor_table(adjacency, sample_count=100):
    row, col = adjacency.indices()
    num_nodes = adjacency.size(0)
    degrees = torch.bincount(row, minlength=num_nodes)
    offsets = torch.cumsum(degrees, dim=0) - degrees
    sample_steps = torch.arange(
        1,
        sample_count + 1,
        device=adjacency.device,
    ).unsqueeze(1)
    positions = offsets.unsqueeze(0) + torch.remainder(
        sample_steps,
        degrees.unsqueeze(0),
    )
    return col[positions]

def make_class_split_masks(labels, args, train_num, val_num):
    num_classes = int(labels.max().item()) + 1
    all_classes = torch.arange(num_classes)

    torch.manual_seed(args.seed)
    shuffled_classes = all_classes[torch.randperm(num_classes)]
    train_classes = shuffled_classes[:train_num]
    val_classes = shuffled_classes[train_num:train_num + val_num]
    test_classes = shuffled_classes[train_num + val_num:]

    print("Dataset class split:")
    print(f"Training classes: {train_classes.tolist()}")
    print(f"Validation classes: {val_classes.tolist()}")
    print(f"Test classes: {test_classes.tolist()}")

    train_mask = torch.isin(labels, train_classes)
    val_mask = torch.isin(labels, val_classes)
    test_mask = torch.isin(labels, test_classes)
    return DatasetMasks(train_mask, val_mask, test_mask)


def make_cora_full_masks(labels, args):
    # Match COLA's CoraFull class split exactly: start from classes sorted by
    # label ID and exclude labels 68 and 69, which contain too few samples.
    remaining_classes = torch.unique(labels).cpu()
    if remaining_classes.numel() == 70:
        remaining_classes = remaining_classes[:-2]

    # Use an independent generator so unrelated random operations cannot
    # change the class split for a given seed.
    generator = torch.Generator().manual_seed(int(args.seed))
    shuffled = remaining_classes[
        torch.randperm(len(remaining_classes), generator=generator)
    ]
    train_classes = shuffled[:38]
    val_classes = shuffled[38:53]
    test_classes = shuffled[53:]
    return DatasetMasks(
        torch.isin(labels, train_classes),
        torch.isin(labels, val_classes),
        torch.isin(labels, test_classes),
    )


def masks_for_run(prepared_graph, args):
    labels = prepared_graph.labels_cpu
    split_sizes = {
        'Cora': (3, 2),
        'CiteSeer': (2, 2),
        'Computers': (4, 3),
        'CS': (5, 5),
    }
    if prepared_graph.dataset_name in split_sizes:
        return make_class_split_masks(
            labels,
            args,
            *split_sizes[prepared_graph.dataset_name],
        )
    if prepared_graph.dataset_name == 'CoraFull':
        return make_cora_full_masks(labels, args)
    raise ValueError(f"Unsupported dataset: {prepared_graph.dataset_name!r}.")


def describe_mask_split(labels, mask):
    """Return an exact, compact description of one dataset partition."""
    labels = labels.cpu()
    mask = mask.cpu().bool()
    selected_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    selected_classes = torch.unique(labels[selected_indices]).tolist()
    selected_classes = sorted(int(value) for value in selected_classes)

    class_mask = torch.isin(
        labels,
        torch.tensor(selected_classes, dtype=labels.dtype),
    )
    if torch.equal(mask, class_mask):
        description = {'type': 'classes', 'values': selected_classes}
    else:
        description = {
            'type': 'nodes',
            'values': selected_indices.tolist(),
        }
    return json.dumps(description, ensure_ascii=False, separators=(',', ':'))


def describe_dataset_split(prepared_graph, masks):
    labels = prepared_graph.labels_cpu
    return tuple(
        describe_mask_split(labels, mask)
        for mask in (masks.train, masks.val, masks.test)
    )


def get_dataset(args):
    if args.dataset_name in ['Cora', 'CiteSeer', 'Computers', 'CoraFull', 'CS']:
        path = str(Path(args.data_root).expanduser().resolve())
        if args.dataset_name in ['Cora', 'CiteSeer']:
            dataset = Planetoid(path, args.dataset_name)
            data = dataset[0]
        elif args.dataset_name == 'Computers':
            dataset = Amazon(path, args.dataset_name, pre_transform=None)  
            data = dataset[0]
        elif args.dataset_name == 'CS':
            dataset = Coauthor(path, name='CS')
            data = dataset[0]
        elif args.dataset_name == 'CoraFull':
            dataset = CitationFull(
                osp.join(path, 'CoraFull'),
                name='Cora',
            )
            data = dataset[0]

        adjacency = build_normalized_sparse_adjacency(
            data.edge_index,
            data.num_nodes,
        )
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))

    return data, adjacency


def prepare_graph(args, running_device):
    cprint("## Loading Dataset ##", "yellow")
    data, adjacency = get_dataset(args)
    labels_cpu = data.y.long().cpu()

    adjacency = adjacency.to(running_device).coalesce()
    prepared_graph = PreparedGraph(
        dataset_name=args.dataset_name,
        features=data.x.to(running_device),
        adjacency=adjacency,
        labels=labels_cpu.to(running_device),
        labels_cpu=labels_cpu,
        neighbor_table=build_neighbor_table(adjacency),
        num_features=data.num_features,
        num_nodes=data.num_nodes,
    )
    cprint("## Done ##", "yellow")
    return prepared_graph


def get_cached_prepared_graph(args, gpu_id, graph_cache):
    running_device = resolve_device(gpu_id)
    cache_key = (args.dataset_name, str(running_device))
    if cache_key not in graph_cache:
        graph_cache[cache_key] = prepare_graph(args, running_device)
    return graph_cache[cache_key]


def select_support_indices(query_embs, candidate_embs, k):
    """Select pseudo supports by cross-branch cosine similarity."""
    if k > candidate_embs.size(0):
        raise ValueError("k exceeds the number of candidate nodes.")
    query_norm = F.normalize(query_embs, p=2, dim=1)
    candidate_norm = F.normalize(candidate_embs, p=2, dim=1)
    scores = torch.mm(query_norm, candidate_norm.t())
    return torch.topk(scores, k=k, dim=1, largest=True).indices


def aggregate_sampled_neighbors(
    node_embeddings,
    topk_indices,
    neighbor_table,
    epoch,
    neighbor_count,
):
    sample_rows = (
        epoch
        + 2 * torch.arange(neighbor_count, device=topk_indices.device)
    ) % neighbor_table.size(0)
    sampled_node_indices = neighbor_table[sample_rows][:, topk_indices]
    return node_embeddings[sampled_node_indices].mean(dim=0)


def compute_batched_distances(
    task_embeddings,
    propagated_embeddings,
    neighbor_embeddings,
    topk_indices,
    negative_indices,
):
    task_for_support = task_embeddings.unsqueeze(1).expand(
        -1,
        topk_indices.size(1),
        -1,
    )
    support_distances = F.pairwise_distance(
        task_for_support,
        propagated_embeddings[topk_indices],
    )
    neighbor_distances = F.pairwise_distance(
        task_for_support,
        neighbor_embeddings,
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
    return support_distances, neighbor_distances, negative_distances


def compute_batched_margin_terms(
    support_distances,
    neighbor_distances,
    negative_distances,
    margin,
    mask_margin,
):
    support_expanded = support_distances.unsqueeze(0).expand(
        negative_distances.size(0),
        -1,
        -1,
    )
    neighbor_expanded = neighbor_distances.unsqueeze(0).expand_as(
        support_expanded
    )
    negative_expanded = negative_distances.unsqueeze(-1).expand_as(
        support_expanded
    )
    margin_labels = -torch.ones_like(support_expanded)

    support_margin = F.margin_ranking_loss(
        support_expanded,
        negative_expanded,
        margin_labels,
        margin=margin,
        reduction='none',
    )
    neighbor_margin = F.margin_ranking_loss(
        neighbor_expanded,
        negative_expanded,
        margin_labels,
        margin=margin,
        reduction='none',
    )
    mask_values = torch.clamp(
        negative_expanded - support_expanded.detach() - mask_margin,
        min=0,
    )
    return (
        support_margin.mean(dim=1).sum(),
        neighbor_margin.mean(dim=1).sum(),
        mask_values.sum(),
    )


def run_CMFC(
    args,
    gpu_id=None,
    prepared_graph=None,
    masks=None,
):
    if args.m <= 0:
        raise ValueError(f"m must be a positive integer, got {args.m}.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = resolve_device(gpu_id)
    if prepared_graph is None:
        prepared_graph = prepare_graph(args, running_device)
    if prepared_graph.dataset_name != args.dataset_name:
        raise ValueError(
            "Prepared graph dataset does not match args.dataset_name: "
            f"{prepared_graph.dataset_name!r} != {args.dataset_name!r}."
        )

    if masks is None:
        masks = masks_for_run(prepared_graph, args)
    nb_feature = prepared_graph.num_features
    nb_nodes = prepared_graph.num_nodes
    feature_X = prepared_graph.features
    A_I_nomal = prepared_graph.adjacency
    
    model = CMFC_Fast(nb_feature, cfg=args.cfg, dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)

    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    num_neg = args.NN
        
    start = t()
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimiser.zero_grad()
        h_a, h_p, h_a2 = model(feature_X, A_I_nomal)
        num_task_node = min(args.n * args.task_num, nb_nodes)

        task_node_index = np.random.choice(nb_nodes, size=num_task_node, replace=False)
        task_node_index = torch.LongTensor(task_node_index).to(running_device)
        
        negative_indices = torch.as_tensor(
            np.stack([
                np.random.permutation(num_task_node)
                for _ in range(num_neg)
            ]),
            dtype=torch.long,
            device=running_device,
        )

        h_a_task = h_a[task_node_index]  

        topk_indices = select_support_indices(
            h_a_task,
            h_a2,
            args.k,
        )
        neighbor_embeddings = aggregate_sampled_neighbors(
            h_a2,
            topk_indices,
            prepared_graph.neighbor_table,
            epoch,
            args.m,
        )

        support_distances, neighbor_distances, negative_distances = (
            compute_batched_distances(
                h_a_task,
                h_p,
                neighbor_embeddings,
                topk_indices,
                negative_indices,
            )
        )

        task_loss_mar, task_loss_mar_1, task_mask_margin_N = (
            compute_batched_margin_terms(
                support_distances,
                neighbor_distances,
                negative_distances,
                my_margin,
                my_margin_2,
            )
        )

        loss_mar = task_loss_mar / (num_task_node * args.k) * 100
        loss_mar_1 = task_loss_mar_1 / (num_task_node * args.k) * 100
        mask_margin_N = task_mask_margin_N / (num_task_node * args.k * num_neg) * 100
         
        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3
        loss.backward()
        optimiser.step()

        if args.save_model:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
            
    result = evaluate_on_test_set(
        model,
        prepared_graph,
        masks.test,
        args,
        "Final test set",
    )
    noe = t()
    print('total time', noe - start)
    return result

def evaluate_on_test_set(
    model,
    prepared_graph,
    test_mask,
    args,
    eval_type="Test set",
):
    model.eval()
    with torch.no_grad():
        _, propagated_embeddings, _ = model.embed(
            prepared_graph.features,
            prepared_graph.adjacency,
        )
        normalized_embeddings = propagated_embeddings / (
            propagated_embeddings.norm(dim=1)[:, None]
        )
        embeddings_np = normalized_embeddings.cpu().numpy()

    eval_indices = torch.nonzero(
        test_mask,
        as_tuple=False,
    ).view(-1).cpu().numpy()
    labels_np = prepared_graph.labels_cpu.numpy()
    eval_labels = labels_np[eval_indices]
    available_classes = np.unique(eval_labels)
    n_way = min(args.n, len(available_classes))
    class_indices_by_label = {
        class_idx: eval_indices[eval_labels == class_idx]
        for class_idx in available_classes
    }

    test_acc_all = []
    n_repeats = 100
    for _ in range(n_repeats):
        selected_classes = np.random.choice(
            available_classes,
            size=n_way,
            replace=False,
        )

        class_supports = []
        class_queries = []
        support_lbls = []
        query_lbls = []

        for i, class_idx in enumerate(selected_classes):
            class_indices = class_indices_by_label[class_idx].copy()
            if len(class_indices) > args.k + args.q:
                np.random.shuffle(class_indices)
                support_idx = class_indices[:args.k]
                query_idx = class_indices[args.k:args.k + args.q]

                class_supports.append(support_idx)
                class_queries.append(query_idx)
                support_lbls.extend([i] * len(support_idx))
                query_lbls.extend([i] * len(query_idx))

        if len(class_supports) < 2:
            continue

        support_idx = np.concatenate(class_supports)
        query_idx = np.concatenate(class_queries)
        support_lbls = np.array(support_lbls)
        query_lbls = np.array(query_lbls)

        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(embeddings_np[support_idx], support_lbls)
        test_acc_all.append(clf.score(embeddings_np[query_idx], query_lbls))

    if test_acc_all:
        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))

        print(
            f"\n{eval_type} results: {n_way}-way {args.k}-shot "
            f"(q={args.q}) accuracy: {final_mean * 100:.2f}"
            f"+/-{final_interval * 100:.2f}%"
        )
        return final_mean * 100
    return None


RESULT_COLUMNS = [
    'record_type',
    'run',
    'seed',
    'train_split',
    'validation_split',
    'test_split',
    'accuracy_percent',
    'mean_accuracy_percent',
    'confidence_interval_95_percent',
]


def result_csv_path(args):
    result_dir = osp.join(
        args.output_dir,
        str(args.dataset_name),
    )
    return osp.join(result_dir, f'{args.n}-way {args.k}-shot.csv')


def initialize_result_csv(csv_path):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        csv.writer(csv_file).writerow(RESULT_COLUMNS)


def append_result_row(csv_path, row):
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as csv_file:
        csv.writer(csv_file).writerow(row)


def run_repeated_experiment(
    args,
    gpu_id,
    graph_cache=None,
    result_path=None,
):
    num_runs = int(args.num_runs)
    if num_runs <= 0:
        raise ValueError(f"num_runs must be a positive integer, got {num_runs}.")

    if graph_cache is None:
        graph_cache = {}
    prepared_graph = get_cached_prepared_graph(args, gpu_id, graph_cache)
    base_seed = int(args.seed)
    run_results = []
    if result_path is not None:
        initialize_result_csv(result_path)

    for run in range(num_runs):
        run_args = copy.deepcopy(args)
        run_seed = base_seed + run * 1000
        run_args.seed = run_seed
        print(f"\nExperiment {run + 1}/{num_runs} | Seed: {run_seed}")

        masks = None
        split_description = None
        if result_path is not None:
            masks = masks_for_run(prepared_graph, run_args)
            split_description = describe_dataset_split(prepared_graph, masks)

        run_result = run_CMFC(
            run_args,
            gpu_id=gpu_id,
            prepared_graph=prepared_graph,
            masks=masks,
        )
        if run_result is None:
            raise RuntimeError(
                f"Experiment {run + 1}/{num_runs} did not return an accuracy."
            )
        run_results.append(run_result)
        if result_path is not None:
            append_result_row(result_path, [
                'run',
                run + 1,
                run_seed,
                *split_description,
                f'{run_result:.6f}',
                '',
                '',
            ])

    mean_accuracy = float(np.mean(run_results))
    confidence_interval = float(
        1.96 * (np.std(run_results) / np.sqrt(len(run_results)))
    )
    if result_path is not None:
        append_result_row(result_path, [
            'summary',
            '',
            '',
            '',
            '',
            '',
            '',
            f'{mean_accuracy:.6f}',
            f'{confidence_interval:.6f}',
        ])
        print(f'Results saved to: {result_path}')
    return mean_accuracy, confidence_interval


if __name__ == '__main__':
    main_args = get_args()

    pprint_args(main_args)
    running_device = resolve_configured_device(main_args)
    print(f"Using device: {running_device}")

    mean_accuracy, confidence_interval = run_repeated_experiment(
        main_args,
        running_device,
        result_path=result_csv_path(main_args),
    )
    print(
        f"\n{'=' * 30}\n"
        f"Final Results over {main_args.num_runs} runs:\n"
        f"Acc: {mean_accuracy:.2f}% +/- {confidence_interval:.2f}%\n"
        f"{'=' * 30}"
    )
