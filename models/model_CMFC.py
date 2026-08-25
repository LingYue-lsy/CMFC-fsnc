import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if adj.layout == torch.strided:
            return torch.mm(adj, embeddings)
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
