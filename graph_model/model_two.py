import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
from dcrnn_model import DCRNNModel

class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=18, output_feat=64,
                 seq_len=15, pred_seq_len=1, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.max_nodes = 1

        self.edge_importance_weighting = True

        self.convolution = nn.Conv2d(512, 18, kernel_size=1, stride=1)
        self.batch = nn.BatchNorm2d(18)
        # self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AvgPool2d(kernel_size=6)

        self.data_bn = nn.BatchNorm1d(input_feat*self.max_nodes)
        self.data_bn_loc = nn.BatchNorm1d(4*self.max_nodes)


    def forward(self, v, a, loc):

        b, t, n, h, w, c = v.size()
        v = v.permute(0, 1, 2, 5, 3, 4).contiguous()
        v = v.view(b*t*n, c, h, w)
        v = self.batch(F.relu(self.convolution(v)))
        v = self.pool((v)).reshape(b*t*n, -1)

        v = v.view(b, t, n, -1)
        b, t, n, c = v.size()
        v = v.permute(0, 3, 2, 1).contiguous()
        v = v.view(b, c * n, t)
        v = self.data_bn(v)
        v = v.view(b, c, n, t)
        v = v.permute(0, 1, 3, 2).contiguous()

        # loc = loc.permute(0, 3, 2, 1).contiguous()
        # loc = loc.view(b, 4*n, t)
        # loc = self.data_bn_loc(loc)
        # loc = loc.view(b, 4, n, t)
        # loc = loc.permute(0, 1, 3, 2).contiguous()

        v = v.permute(0, 2, 1, 3).contiguous()
        v = v.view(b * t, -1, n)
        v = F.avg_pool1d(v, v.size()[2])
        v = v.reshape(b, t, -1)

        loc = loc.reshape(b, t, -1)

        x = torch.cat((v, loc), dim=2)
        x = self.classifier(x)
        # x, _ = self.dec(x)
        # x = torch.tanh(self.dropout_dec(x))
        # print(x.shape)
        # x = self.fcn(x[:, -1])
        x = self.fcn(x.view(b, -1))
        x = x.view(x.size(0), -1)

        return x, a
