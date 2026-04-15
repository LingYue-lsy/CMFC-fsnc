from time import perf_counter as t
import random
import torch
from torch_geometric.utils import to_undirected
import os.path as osp
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from scipy.sparse import csr_matrix
from data_unit.utils import blind_other_gpus, normalize_graph
from models import CMFC_Fast
from torch_geometric.utils import degree
import os
from sklearn.linear_model import LogisticRegression
from termcolor import cprint
from args_utils import get_args, pprint_args

# 类别划分
def split_class(data, args, train_num, val_num):
    num_nodes = data.x.size(0)
    num_classes = int(data.y.max().item()) + 1
            
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool) 
             
    all_classes = torch.arange(num_classes)
            
    torch.manual_seed(args.seed)
    shuffled_classes = all_classes[torch.randperm(num_classes)]
            
    train_classes = shuffled_classes[:train_num]  
    val_classes = shuffled_classes[train_num:train_num + val_num]   
    test_classes = shuffled_classes[train_num + val_num:]    
            
    print(f"数据集类别划分:")
    print(f"训练集类别: {train_classes.tolist()}")
    print(f"验证集类别: {val_classes.tolist()}")
    print(f"测试集类别: {test_classes.tolist()}")
            
    for class_idx in train_classes:
        class_indices = torch.where(data.y == class_idx)[0]
        train_mask[class_indices] = True
            
    for class_idx in val_classes:
        class_indices = torch.where(data.y == class_idx)[0]
        val_mask[class_indices] = True
            
    for class_idx in test_classes:
        class_indices = torch.where(data.y == class_idx)[0]
        test_mask[class_indices] = True
            
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

# 数据集加载
def get_dataset(args, dataset_kwargs):
    if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'CoraFull', 'CS']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/')
        if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(path, args.dataset_name)
            data = dataset[0]
            if args.dataset_name == 'Cora':
                split_class(data, args, 3, 2)
            elif args.dataset_name == 'CiteSeer':
                split_class(data, args, 2, 2)
        elif args.dataset_name == 'Photo':
            dataset = Amazon(path, args.dataset_name, pre_transform=None) 
            data = dataset[0] 
        elif args.dataset_name == 'Computers':
            dataset = Amazon(path, args.dataset_name, pre_transform=None)  
            data = dataset[0]
            split_class(data, args, 4, 3)
        elif args.dataset_name == 'CS':
            dataset = Coauthor(path, name='CS')
            data = dataset[0]
            split_class(data, args, 5, 5)
        elif args.dataset_name == 'CoraFull':
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/CoraFull/cora')
            raw_path = osp.join(path, 'raw')
            data_file = osp.join(raw_path, 'cora.npz')
            if osp.exists(data_file):
                loaded_data = np.load(data_file)
                adj_matrix = csr_matrix((loaded_data['adj_data'], 
                                       loaded_data['adj_indices'], 
                                       loaded_data['adj_indptr']), 
                                      shape=loaded_data['adj_shape'])
                attr_matrix = csr_matrix((loaded_data['attr_data'], 
                                         loaded_data['attr_indices'], 
                                         loaded_data['attr_indptr']), 
                                        shape=loaded_data['attr_shape'])
                x = torch.FloatTensor(attr_matrix.toarray())
                edge_index = torch.LongTensor(np.vstack(adj_matrix.nonzero()))
                y = torch.LongTensor(loaded_data['labels'])
                data = Data(x=x, edge_index=edge_index, y=y)
                
                num_nodes = data.x.size(0)
                num_classes = int(data.y.max().item()) + 1
                
                class_counts = []
                for i in range(num_classes):
                    class_indices = torch.where(data.y == i)[0]
                    class_counts.append((i, len(class_indices)))
                
                class_counts.sort(key=lambda x: x[1])
                excluded_classes = [class_counts[0][0], class_counts[1][0]]  
                remaining_classes = [x[0] for x in class_counts[2:]]  
                
                torch.manual_seed(args.seed)
                shuffled_remaining = torch.tensor(remaining_classes)[torch.randperm(len(remaining_classes))]
                
                train_classes = shuffled_remaining[:38].tolist()  
                val_classes = shuffled_remaining[38:53].tolist()  
                test_classes = shuffled_remaining[53:].tolist()  
                
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                for class_idx in train_classes:
                    class_indices = torch.where(data.y == class_idx)[0]
                    train_mask[class_indices] = True
                for class_idx in val_classes:
                    class_indices = torch.where(data.y == class_idx)[0]
                    val_mask[class_indices] = True
                for class_idx in test_classes:
                    class_indices = torch.where(data.y == class_idx)[0]
                    test_mask[class_indices] = True
                
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask

        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        edge_index_np = data.edge_index.numpy()
        i = torch.LongTensor(np.array([edge_index_np[0], edge_index_np[1]]))
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        A = A_sp.to_dense()
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        A_I_nomal = normalize_graph(A_I)
        A_I_nomal = A_I_nomal.to_sparse()

        lable = data.y
        nb_feature = data.num_features
        nb_classes = int(lable.max() - lable.min()) + 1
        nb_nodes = data.num_nodes
        data.x = torch.FloatTensor(data.x)
        eps = 2.2204e-16
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj_1 = csr_matrix(
            (np.ones(data.num_edges), (edge_index_np[0], edge_index_np[1])),
            shape=(data.num_nodes, data.num_nodes))

    return data, [A_I_nomal,adj_1], [data.x], [lable, nb_feature, nb_classes, nb_nodes]


def run_CMFC(args, gpu_id=None, run_id=0, **kwargs):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    cprint("## Loading Dataset ##", "yellow")
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, dataset_kwargs)
    lable = nb_list[0]
    nb_feature = nb_list[1]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    cprint("## Done ##", "yellow")
    
    model = CMFC_Fast(nb_feature, cfg=args.cfg, dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)
    lable = lable.to(running_device)

    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    edge_index = A_I_nomal._indices()[1]

    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduction='none')
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(running_device)
    
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
        
    idx_p_list = []
    for j in range(1, 101):
        random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
        idx_p = edge_index[random_list]
        idx_p_list.append(idx_p)
        
    # 训练过程
    start = t()
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        model.train()
        optimiser.zero_grad()
        h_a, h_p, h_a2 = model(feature_X, A_I_nomal)
        num_task_node = args.n * args.task_num

        task_node_index = np.random.choice(nb_nodes, size=min(num_task_node, nb_nodes), replace=False)
        task_node_index = torch.LongTensor(task_node_index).to(running_device)
        
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(num_task_node)
            idx_list.append(idx_0)

        h_a_task = h_a[task_node_index]  

        # 计算 Top-k 邻居用于构建正样本对
        h_a_task_norm = F.normalize(h_a_task, p=2, dim=1)
        h_a2_norm = F.normalize(h_a2, p=2, dim=1)
        similarity_matrix = torch.mm(h_a_task_norm, h_a2_norm.t())
        _, topk_indices = torch.topk(similarity_matrix, k=args.k, dim=1, largest=True)
        topk_index_list = [indices.cpu().numpy() for indices in topk_indices]

        # 计算邻居表征 (h_nei)
        h_nei = []  
        for i, topk_index in enumerate(topk_index_list):
            topk_index_tensor = torch.tensor(topk_index, device=running_device)
            neighbors_1 = h_a2[idx_p_list[epoch % 100][topk_index_tensor]]
            neighbors_2 = h_a2[idx_p_list[(epoch + 2) % 100][topk_index_tensor]]
            neighbors_3 = h_a2[idx_p_list[(epoch + 4) % 100][topk_index_tensor]]
            neighbors_4 = h_a2[idx_p_list[(epoch + 6) % 100][topk_index_tensor]]
            neighbors_5 = h_a2[idx_p_list[(epoch + 8) % 100][topk_index_tensor]]  
            nei_emb = (neighbors_1 + neighbors_2 + neighbors_3 + neighbors_4 + neighbors_5) / 5  
            h_nei.append(nei_emb)

        nei_d_list = []  
        s_d_list = []  

        for i, node_idx in enumerate(task_node_index):
            task_node_a = h_a_task[i].unsqueeze(0)  
            topk_index = topk_index_list[i]
            topk_nodes_p = h_p[torch.tensor(topk_index, device=running_device)] 
            s_d = F.pairwise_distance(task_node_a.expand_as(topk_nodes_p), topk_nodes_p)  
            s_d_list.append(s_d)
            topk_nodes_nei = h_nei[i] 
            nei_d = F.pairwise_distance(task_node_a.expand_as(topk_nodes_nei), topk_nodes_nei) 
            nei_d_list.append(nei_d)

        neg_d_list = []
        for h_n in idx_list:
            neg_d = F.pairwise_distance(h_a_task, h_a_task[h_n])
            neg_d_list.append(neg_d)

        s_d_stack = torch.stack(s_d_list, dim=0)  
        nei_d_stack = torch.stack(nei_d_list, dim=0)  

        task_loss_mar = 0
        task_loss_mar_1 = 0
        task_mask_margin_N = 0
        
        for neg_d in neg_d_list:  
            for j in range(s_d_stack.shape[1]):  
                s_d = s_d_stack[:, j]  
                nei_d = nei_d_stack[:, j]  
                margin_label = -1 * torch.ones_like(s_d)
                task_loss_mar += (margin_loss(s_d, neg_d, margin_label)).mean()
                task_loss_mar_1 += (margin_loss(nei_d, neg_d, margin_label)).mean()
                task_mask_margin_N += torch.max((neg_d - s_d.detach() - my_margin_2), lbl_z).sum()

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
            
        # 最终测试
        if epoch % args.epochs == 0 and epoch != 0:
            result = evaluate_on_test_set(model, feature_X, A_I_nomal, data, lable, args, running_device, "最终测试集")
            noe = t()
            print('total time', noe - start)
            return result

    noe = t()
    print('total time', noe - start)
    return None

# 测试评估
def evaluate_on_test_set(model, feature_X, A_I_nomal, data, lable, args, running_device, eval_type="测试集"):
    model.eval()
    with torch.no_grad():
        h_a, h_p, h_a2 = model.embed(feature_X, A_I_nomal)
        embs = h_p
        embs = embs / embs.norm(dim=1)[:, None]  
        
        test_acc_all = []
        n_repeats = 100
        
        for _ in range(n_repeats):
            eval_indices = torch.nonzero(data.test_mask).squeeze().cpu().numpy()
            eval_labels = lable[eval_indices].cpu().numpy()
            
            available_classes = np.unique(eval_labels)
            n_way = min(args.n, len(available_classes))
            selected_classes = np.random.choice(available_classes, size=n_way, replace=False)
            
            class_supports = []
            class_queries = []
            support_lbls = []
            query_lbls = []
            
            for i, class_idx in enumerate(selected_classes):
                class_eval_mask = (eval_labels == class_idx)
                class_indices = eval_indices[class_eval_mask]
                
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
            
            support_embs = embs[support_idx].cpu().numpy()
            query_embs = embs[query_idx].cpu().numpy()
            
            clf = LogisticRegression(solver='lbfgs', max_iter=1000)
            clf.fit(support_embs, support_lbls)
            test_acc = clf.score(query_embs, query_lbls)
            test_acc_all.append(test_acc)
        
        if len(test_acc_all) > 0:
            final_mean = np.mean(test_acc_all)
            final_std = np.std(test_acc_all)
            final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))
            
            print(f"\n{eval_type}评估结果: {n_way}-way {args.k}-shot (q={args.q}) 准确率: {final_mean * 100:.2f}±{final_interval * 100:.2f}%")
            return final_mean * 100
        return None

if __name__ == '__main__':
    main_args = get_args(model_name="CMFC", dataset_class="Planetoid", dataset_name="Cora", custom_key="classification")
    
    main_args.num_gpus_total = 1  
    main_args.black_list = []     
    main_args.num_gpus_to_use = 1 

    pprint_args(main_args)

    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     black_list=main_args.black_list)
        if not alloc_gpu: alloc_gpu = [0]  

    num_runs = main_args.num_runs
    all_run_results = []
    
    for run in range(num_runs):
        run_seed = main_args.seed + run * 1000  
        main_args.seed = run_seed
        print(f"\nExperiment {run + 1}/{num_runs} | Seed: {run_seed}")

        run_result = run_CMFC(main_args, gpu_id=alloc_gpu[0], run_id=run)
        if run_result is not None:
            all_run_results.append(run_result)
    
    if all_run_results:
        print(f"\n{'='*30}\nFinal Results over {len(all_run_results)} runs:\nAcc: {np.mean(all_run_results):.2f}% ± {1.96 * (np.std(all_run_results) / np.sqrt(len(all_run_results))):.2f}%\n{'='*30}")