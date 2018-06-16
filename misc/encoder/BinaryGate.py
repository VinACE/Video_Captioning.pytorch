import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torchvision.models as models

class BinaryGate(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        if inplace:
            output = input
        else:
            output = input.clone()
        ctx.thrs = random.uniform(0, 1) if training else 0.5
        output[output > ctx.thrs] = 1
        output[output <= ctx.thrs] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None