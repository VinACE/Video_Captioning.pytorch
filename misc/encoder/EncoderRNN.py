import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torchvision.models as models
from misc.encoder.BoundaryDetector import *

class EncoderRNN(nn.Module):
    '''
    Hierarchical Boundart-Aware video encoder
    '''

    def __init__(self, opt):
        #feature_size, projected_size, mid_size, hidden_size, max_frames
        super(EncoderRNN, self).__init__()

        self.feature_size = opt.frame_feat_size
        self.hidden_size = opt.hidden_size
        self.max_frames = opt.max_frames

        self.frame_embed = nn.Linear(opt.frame_feat_size, opt.hidden_size)
        self.frame_drop = nn.Dropout(p=opt.drop_prob_lm)

        self.lstm1_cell = nn.LSTMCell(opt.hidden_size, opt.hidden_size)
        self.lstm1_drop = nn.Dropout(p=opt.drop_prob_lm)

        self.bd = BoundaryDetector(opt.hidden_size, opt.hidden_size, opt.mid_size)

        self.lstm2_cell = nn.LSTMCell(opt.hidden_size, opt.hidden_size, bias=False)
        self.lstm2_drop = nn.Dropout(p=opt.drop_prob_lm)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return Variable(d.data.new(bsz, self.hidden_size).zero_()), \
            Variable(d.data.new(bsz, self.hidden_size).zero_())

    def forward(self, video_feats):
        batch_size = len(video_feats)
        lstm1_h, lstm1_c = self._init_lstm_state(video_feats)
        lstm2_h, lstm2_c = self._init_lstm_state(video_feats)
        video_feats = video_feats[:, :, :self.feature_size].contiguous()
        v = video_feats.view(-1, self.feature_size)
        v = F.relu(self.frame_embed(v))
        v = self.frame_drop(v)
        v = v.view(batch_size, -1, self.hidden_size)
        for i in range(self.max_frames):
            s = self.bd(v[:, i, :], lstm1_h)
            lstm1_h, lstm1_c = self.lstm1_cell(v[:, i, :], (lstm1_h, lstm1_c))
            lstm1_h = self.lstm1_drop(lstm1_h)
            lstm2_input = lstm1_h * s
            lstm2_h, lstm2_c = self.lstm2_cell(lstm2_input, (lstm2_h, lstm2_c))
            lstm2_h = self.lstm2_drop(lstm2_h)
            lstm1_h = lstm1_h * (1 - s)
            lstm1_c = lstm1_c * (1 - s)
        return lstm2_h