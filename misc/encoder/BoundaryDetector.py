import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torchvision.models as models
from misc.encoder.BinaryGate import *

class BoundaryDetector(nn.Module):
    def __init__(self, i_features, h_features, s_features, inplace=False):
        super(BoundaryDetector, self).__init__()
        self.inplace = inplace
        self.Wsi = Parameter(torch.Tensor(s_features, i_features))
        self.Wsh = Parameter(torch.Tensor(s_features, h_features))
        self.bias = Parameter(torch.Tensor(s_features))
        self.vs = Parameter(torch.Tensor(1, s_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Wsi.size(1))
        self.Wsi.data.uniform_(-stdv, stdv)
        self.Wsh.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.vs.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        z = F.linear(x, self.Wsi) + F.linear(h, self.Wsh) + self.bias
        z = F.sigmoid(F.linear(z, self.vs))
        return BinaryGate.apply(z, self.training, self.inplace)

    def __repr__(self):
        return self.__class__.__name__