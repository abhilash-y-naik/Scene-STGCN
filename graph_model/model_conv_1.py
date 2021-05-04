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

class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            # padding=(t_padding, 0),
            # stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x, A):

        assert self.kernel_size == A.size(1)

        b, t, n, c, h, w = x.size()
        x = self.conv(x.view(b*t*n, c, h, w))
        _, c, h, w = x.size()
        x = x.view(b, t, n, c, h, w)
        x = torch.einsum('ntvchl,ntvw->ntwchl', (x, A))

        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 width,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        # print('tcn',kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels*width*width),
            nn.PReLU(),
            nn.Conv2d(
                out_channels*width*width,
                out_channels*width*width,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels*width*width),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        # elif (in_channels == out_channels) and (stride == 1):
        #     self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels*(width+1)*(width+1),
                    out_channels*(width)*(width),
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels*(width)*(width)),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        b, t, n, c, h, w = x.size()

        x_res = x.permute(0, 3, 4, 5, 1, 2).contiguous()
        x_res = x_res.view(b, c*h*w, t, n)

        res = self.residual(x_res)
        x, A = self.gcn(x, A)

        b, t, n, c, h, w = x.size()
        x = x.permute(0, 3, 4, 5, 1, 2).contiguous()
        x = x.view(b, c*h*w, t, n)
        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        x = x.view(b, -1, h, w, t, n)
        x = x.permute(0, 4, 5, 1, 2, 3).contiguous()
        # print(x.shape)
        return x, A


class gcn_spatial(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(gcn_spatial, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        # assert A.size(1) == self.kernel_size
        b, n, c, h, w = x.size()
        x = self.conv(x.view(b*n, c, h, w))
        _, c, h, w = x.size()
        x = x.view(b, n, c, h, w)

        x = torch.einsum('nvchl,nvw->nwchl', (x, A))

        # x = self.batchnorm(x.view(b*n, c, h, w))
        # x = x.view(b, n, c, h, w)

        return x.contiguous(), A



class social_stgcnn(nn.Module):
    def __init__(self, max_nodes=1, n_stgcnn=1, n_txpcnn=1, input_feat=512, output_feat=16,
                 seq_len=15, pred_seq_len=1, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.max_nodes = max_nodes
        self.seq_len =seq_len
        self.edge_importance_weighting = True

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(input_feat, output_feat, (kernel_size, seq_len), width=6, residual=False, dropout=0.5),
            # st_gcn(output_feat, output_feat, (kernel_size, seq_len), width=5, dropout=0.5),
            # st_gcn(output_feat, output_feat, (kernel_size, seq_len), width=4, dropout=0.5)
        ))
        self.seed = 12
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # initialize parameters for edge importance weighting
        if self.edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(seq_len, self.max_nodes, self.max_nodes, requires_grad=True))
                for i in self.st_gcn_networks
            ])

        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # self.gcn_network = nn.ModuleList((
        #         gcn_spatial(16, 16, 15),
        #     ))

        # # initialize parameters for edge importance weighting
        # if self.edge_importance_weighting:
        #     self.edge_importance_gcn = nn.ParameterList([
        #         nn.Parameter(torch.ones(self.max_nodes, self.max_nodes, requires_grad=True))
        #         for i in self.gcn_network
        #     ])
        #
        # else:
        #     self.edge_importance = [1] * len(self.st_gcn_networks)
        # self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dec = nn.LSTM(580, 128, num_layers=1, bias=True,
                           batch_first=True, bidirectional=False)

        nn.init.xavier_normal_(self.dec.weight_hh_l0)
        self.dropout_dec = nn.Dropout(p=0.4)

        self.fcn = nn.Linear(128, pred_seq_len)
        nn.init.xavier_normal_(self.fcn.weight)

    def forward(self, v, a, a_space, loc):

        for graph, importance in zip(self.st_gcn_networks, self.edge_importance):
            v, _ = graph(v, a * importance)

        b, t, n, c, h, w = v.size()
        v = v.permute(0, 3, 4, 5, 1, 2).contiguous()

        # If the sequence information is utilised
        v = v.view(b, -1, t, n)
        v = F.avg_pool2d(v, v.size()[2:])

        # v = v.view(b, c, h, w)

        # for graph, importance in zip(self.gcn_network, self.edge_importance_gcn):
        #     output, _ = graph(v, a_space * importance)
        # _, _, c, h, w = output.size()
        # output = output.permute(0, 2, 3, 4, 1).contiguous()
        # output = output.view(b, c * h * w, n)
        # output = F.avg_pool1d(output, output.size()[2])

        output = self.flatten(v)

        repeat_vec = output.repeat(1, self.seq_len).reshape(b, self.seq_len, -1)
        loc = loc.reshape(b, t, -1)
        x = torch.cat((repeat_vec, loc), dim=2)

        x, _ = self.dec(x)
        x = torch.tanh(self.dropout_dec(x))

        x = self.fcn(x[:, -1])

        x = x.view(x.size(0), -1)

        return x