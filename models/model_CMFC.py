import torch
import torch.nn as nn
import torch.nn.functional as F



def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer=None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer is not None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)

class CMFC_Fast(nn.Module):
    def __init__(self, n_in, cfg=None, dropout=0.2, momentum=0.99):
        super().__init__()
        self.MLP = make_mlplayers(n_in, cfg)  # 第一个MLP，正常更新权重
        self.MLP_ema = make_mlplayers(n_in, cfg)  # 第二个MLP，权重由第一个MLP的移动平均得到
        self.momentum = momentum
        self.dropout = dropout
        
        for m in self.modules():
            self.weights_init(m)
            
        self._copy_weights()
        
        # EMA MLP不参与梯度更新
        self.MLP_ema.requires_grad_(False)

    @torch.no_grad()
    def _copy_weights(self):
        """初始化时复制主MLP的权重到EMA MLP"""
        for param_q, param_k in zip(self.MLP.parameters(), self.MLP_ema.parameters()):
            param_k.copy_(param_q)
    
    @torch.no_grad()
    def _update_ema_weights(self):
        """更新EMA MLP的权重"""
        for param_q, param_k in zip(self.MLP.parameters(), self.MLP_ema.parameters()):
            param_k.mul_(self.momentum)
            param_k.add_(param_q, alpha=1.0 - self.momentum)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _aggregate(adj, embeddings):
        if adj.layout == torch.strided:
            return torch.mm(adj, embeddings)
        return torch.sparse.mm(adj, embeddings)

    def forward(self, seq_a, adj=None):
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        # 第一个MLP：正常更新权重
        h_a = self.MLP(seq_a)
        
        # 更新EMA MLP的权重
        if self.training:
            self._update_ema_weights()
        
        # 使用EMA MLP进行图卷积（GCN与EMA MLP共享权重）
        with torch.no_grad():
            h_a_ema = self.MLP_ema(seq_a)
            h_p = self._aggregate(adj, h_a_ema)
        
        return h_a, h_p, h_a_ema

    def embed(self, seq_a, adj=None):
        # 推理时使用EMA MLP
        with torch.no_grad():
            h_a_ema = self.MLP_ema(seq_a)
            h_p = self._aggregate(adj, h_a_ema)
            
        return h_a_ema, h_p, h_a_ema
