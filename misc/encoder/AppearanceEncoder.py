import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torchvision.models as models

import misc.encoder.resnet as resnet
from misc.encoder.resnet_utils import myResnet

class AppearanceEncoder(nn.Module):
    def __init__(self, opt):
        super(AppearanceEncoder, self).__init__()
        self.resnet = models.resnet101()
        self.resnet.load_state_dict(torch.load(opt.resnet_checkpoint))
        self.my_resnet = myResnet(self.resnet)
        del self.resnet.fc

    def bz_forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def s_forward(self, x):
        tmp_fc, tmp_att = self.my_resnet(x)
        return tmp_fc

    def forward(self, x, resize=True):
        fc = self.bz_forward(x) if resize else self.s_forward(x)
        return fc