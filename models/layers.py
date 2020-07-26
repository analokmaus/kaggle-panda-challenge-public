import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x): 
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)


class ChannelPool(nn.Module):

    def __init__(self, dim=1, concat=True):
        super().__init__()
        self.dim = dim
        self.concat = concat
    
    def forward(self, x):
        max_out = torch.max(x, self.dim)[0].unsqueeze(1)
        avg_out = torch.mean(x, self.dim).unsqueeze(1)
        if self.concat:
            return torch.cat((max_out, avg_out), dim=self.dim)
        else:
            return max_out, avg_out


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1, 1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
