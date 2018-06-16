import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torchvision.models as models
from misc.encoder.C3D import *

class MotionEncoder(nn.Module):

    def __init__(self, opt):
        super(MotionEncoder, self).__init__()
        self.c3d = C3D()
        pretrained_dict = torch.load(opt.c3d_checkpoint)
        model_dict = self.c3d.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.c3d.load_state_dict(model_dict)

    def forward(self, x):
        return self.c3d(x)