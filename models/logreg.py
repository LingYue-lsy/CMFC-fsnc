import torch
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super().__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
