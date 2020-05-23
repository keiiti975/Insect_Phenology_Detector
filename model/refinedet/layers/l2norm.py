import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self,n_channels):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = 10
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.init_parameters()

    def init_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
