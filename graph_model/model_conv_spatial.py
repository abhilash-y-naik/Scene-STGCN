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
from graph_model.convlstm2D import ConvLSTM2D

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
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
        super(ConvTemporalGraphical,self).__init__()
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
        # b, n, c, h, w = x.size()
        # x = self.conv(x.view(b*n, c, h, w))
        # _, c, h, w = x.size()
        # x = x.view(b, n, c, h, w)

        x = torch.einsum('nvchl,nvw->nwchl', (x, A))

        # x = self.batchnorm(x.view(b*n, c, h, w))
        # x = x.view(b, n, c, h, w)

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
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        # res = self.residual(x)
        x, A = self.gcn(x, A)

        # x = self.tcn(x) + res

        # if not self.use_mdn:
        #     x = self.prelu(x)

        return x, A

# class social_stgcnn(nn.Module):
#     def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=512, output_feat=64,
#                  seq_len=15, pred_seq_len=1, kernel_size=15):
#         super(social_stgcnn, self).__init__()
#         self.n_stgcnn = n_stgcnn
#         self.n_txpcnn = n_txpcnn
#         self.max_nodes = 1
#         self.seq_len = seq_len
#         self.edge_importance_weighting = True
#
#         self.encoder_model = ConvLSTM2D(512, 64, dropout=0.4, kernel_size=(2, 2), num_layers=1,
#                                         batch_first=True, bias=True, return_all_layers=False)
#         self.conv = nn.Conv2d(64, 64,
#                               kernel_size=2, bias=True)
#         nn.init.xavier_normal_(self.conv.weight)
#         self.batch = nn.BatchNorm2d(64)
#         self.dropout = nn.Dropout(p=0.4)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.encoder_output = nn.Flatten()
#
#         # self.data_bn = nn.BatchNorm1d(2308*self.max_nodes)
#
#         # self.st_gcn_networks = nn.ModuleList((
#         #     st_gcn(64, 64, (kernel_size, seq_len), 1, residual=False, dropout=0.5),
#         #     st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(128, 128, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(128, 128, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(128, 256, (kernel_size, seq_len), 2, dropout=0.5),
#             # st_gcn(256, 256, (kernel_size, seq_len), 1, dropout=0.5),
#             # st_gcn(256, 256, (kernel_size, seq_len), 1, dropout=0.5),
#         # ))
#
#         # # initialize parameters for edge importance weighting
#         # if self.edge_importance_weighting:
#         #     self.edge_importance = nn.ParameterList([
#         #         nn.Parameter(torch.ones(self.max_nodes, self.max_nodes, requires_grad=True))
#         #         for i in self.st_gcn_networks
#         #     ])
#         #
#         # else:
#         #     self.edge_importance = [1] * len(self.st_gcn_networks)
#
#         self.decoder_model = nn.LSTM(2308, 128,
#                                      bias=True, batch_first=True)
#         nn.init.xavier_normal_(self.decoder_model.weight_hh_l0)
#         self.dropout_lstm = nn.Dropout(p=0.4)
#
#         self.decoder_dense_output = nn.Linear(128, pred_seq_len, bias=True)
#         nn.init.xavier_normal_(self.decoder_dense_output.weight)
#
#     def forward(self, v, a, loc):
#
#         B, T, N, H, W, C = v.size()
#         v = v.view((B, T, N, C, H, W))
#         output = v.permute(0, 2, 1, 3, 4, 5).contiguous()
#         output = output.view(B*N, T, C, H, W)
#         # print(output.shape)
#         _, output = self.encoder_model(output)
#         output = self.pool(output)
#         _, C, H, W = output.size()
#         # output = output.view(B, N, C, H, W)
#
#         # for graph, importance in zip(self.st_gcn_networks, self.edge_importance):
#         #     output, _ = graph(output, a * importance)
#         #
#         # output = output.permute(0, 2, 3, 4, 1).contiguous()
#         # output = output.view(B, C*H*W, N)
#         # output = F.avg_pool1d(output, output.size()[2])
#
#         output = self.encoder_output(output)
#         repeat_vec = output.repeat(1, self.seq_len).view(B, T, -1)
#         output = torch.cat((repeat_vec, loc), dim=2)
#
#         output, _ = self.decoder_model(output)
#         output = self.dropout_lstm(output)
#         output = self.decoder_dense_output(torch.tanh(output[:, -1]))
#
#         output = output.view(output.size(0), -1)
#
#         return output, a

class social_stgcnn(nn.Module):
    '''
    Create an LSTM Encoder-Decoder model for intention estimation
    '''

    def __init__(self,
                    max_nodes, node_info):
        super(social_stgcnn, self).__init__()

        # Learning rate params
        # self.lr_params = learning_rate_params
        self.node_info = node_info
        # Network parameters
        self._num_hidden_units = 128

        # Encoder
        self._lstm_dropout = int(0.4)

        # conv unit parameters
        self._convlstm_num_filters = 64
        self._convlstm_kernel_size = 2

        # decoder unit parameters
        self._decoder_dense_output_size = 1
        self._decoder_input_size = 4  # decided on run time according to data
        self._decoder_seq_length = 15

        # Generate Encoder LSTM Unit
        self.encoder_model = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False)

        # self.encoder_model_ped = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
        #          batch_first=True, bias=True, return_all_layers=False)

        # self.encoder_model_veh = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
        #          batch_first=True, bias=True, return_all_layers=False)

        # self.encoder_model_traf = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
        #          batch_first=True, bias=True, return_all_layers=False)

        # self.encoder_model_egoveh = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
        #          batch_first=True, bias=True, return_all_layers=False)

        self.conv = nn.Conv2d(self._convlstm_num_filters, self._convlstm_num_filters,
                              kernel_size=self._convlstm_kernel_size, bias=True)
        nn.init.xavier_normal_(self.conv.weight)
        self.batch = nn.BatchNorm2d(self._convlstm_num_filters)
        self.dropout = nn.Dropout(p=self._lstm_dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(64, 64, (3, 15), 1, residual=False, dropout=0.5),
        #     st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(64, 64, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(128, 128, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(128, 128, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(128, 256, (kernel_size, seq_len), 2, dropout=0.5),
        # st_gcn(256, 256, (kernel_size, seq_len), 1, dropout=0.5),
        # st_gcn(256, 256, (kernel_size, seq_len), 1, dropout=0.5),
        #     ))
        #
        # self.edge_importance_weighting = True
        # self.max_nodes = max_nodes
        # initialize parameters for edge importance weighting
        # if self.edge_importance_weighting:
        #     self.edge_importance = nn.ParameterList([
        #         nn.Parameter(torch.ones(self.max_nodes, self.max_nodes, requires_grad=True))
        #         for i in self.st_gcn_networks
        #     ])
        #
        # else:
        #     self.edge_importance = [1] * len(self.st_gcn_networks)

        self.encoder_output = nn.Flatten()
        # self.batch_norm = nn.BatchNorm1d(2308)
        # Generate Decoder LSTM unit
        self.decoder_model = nn.LSTM(2308, self._num_hidden_units,
                                     bias=True, batch_first=True)
        nn.init.xavier_normal_(self.decoder_model.weight_hh_l0)
        self.dropout_lstm = nn.Dropout(p=self._lstm_dropout)

        self.decoder_dense_output = nn.Linear(self._num_hidden_units, self._decoder_dense_output_size, bias=True)
        nn.init.xavier_normal_(self.decoder_dense_output.weight)

    # ['pedestrian', 'vehicle', 'traffic_light', 'transit_station', 'sign', 'crosswalk', 'ego_vehicle']
    def forward(self, encoder_input, a, decoder_input):

        B, T, N, C, H, W = encoder_input.size()
        encoder_input = encoder_input.permute(0, 2, 1, 3, 4, 5).contiguous()
        # num_final = num + self.node_info['pedestrian']
        # b, n, T, C, H, W = encoder_input.size()
        output = encoder_input.view(B*N, T, C, H, W)
        _, output = self.encoder_model(output)
        output = self.pool(output)
        _, C, H, W = output.size()
        output = output.view(B, N, C, H, W)

        # num = 0
        # num_final = num + self.node_info['pedestrian']
        # seperate_x = encoder_input[:, num:num_final].contiguous()
        # b, n, T, C, H, W = seperate_x.size()
        # output = seperate_x.view(b*n, T, C, H, W)
        # _, output = self.encoder_model_ped(output)
        # output = self.pool(output)
        # _, C, H, W = output.size()
        # output_ped = output.view(b, n, C, H, W)
        #
        # if self.node_info['vehicle'] > 0:
        #     num = num_final
        #     num_final = num+self.node_info['vehicle']
        #     seperate_x = encoder_input[:, num:num_final].contiguous()
        #     b, n, T, C, H, W = seperate_x.size()
        #     output = seperate_x.view(b*n, T, C, H, W)
        #     _, output = self.encoder_model_veh(output)
        #     output = self.pool(output)
        #     _, C, H, W = output.size()
        #     output_veh = output.view(b, n, C, H, W)
        #     output_ped = torch.cat((output_ped, output_veh), dim=1)
        #
        # if self.node_info['crosswalk'] > 0 or self.node_info['sign'] > 0 \
        # or self.node_info['transit_station'] or self.node_info['traffic_light']:
        #     num = num_final
        #     num_final = num+self.node_info['crosswalk']+ self.node_info['sign']+\
        #                     self.node_info['transit_station']+self.node_info['traffic_light']
        #
        #     seperate_x = encoder_input[:, num: num_final].contiguous()
        #
        #     b, n, T, C, H, W = seperate_x.size()
        #     output = seperate_x.view(b*n, T, C, H, W)
        #     _, output = self.encoder_model_traf(output)
        #     output = self.pool(output)
        #     _, C, H, W = output.size()
        #     output_traf = output.view(b, n, C, H, W)
        #     output_ped = torch.cat((output_ped, output_traf), dim=1)
        #
        # if self.node_info['ego_vehicle'] == 1:
        #
        #     num_final = -1
        #     seperate_x = encoder_input[:, num_final:].contiguous()
        #     b, n, T, C, H, W = seperate_x.size()
        #     output = seperate_x.view(b*n, T, C, H, W)
        #     _, output = self.encoder_model_egoveh(output)
        #     output = self.pool(output)
        #     _, C, H, W = output.size()
        #     output_ego = output.view(b, n, C, H, W)
        #     output_ped = torch.cat((output_ped, output_ego), dim=1)

        # output = output_ped

        # for graph, importance in zip(self.st_gcn_networks, self.edge_importance):
        #     output, _ = graph(output, a * importance)
        #
        # output = output.permute(0, 2, 3, 4, 1).contiguous()
        # output = output.view(B, C * H * W, N)
        # output = F.avg_pool1d(output, output.size()[2])
        # output = torch.sum(output, dim=2)
        # output = output[:, 0]

        # output = output.view(B, N * C * H * W)
        output = self.encoder_output(output)
        repeat_vec = output.repeat(1, self._decoder_seq_length).reshape(B, self._decoder_seq_length, -1)
        output = torch.cat((repeat_vec, decoder_input), dim=2)

        # output = self.batch_nm(output)

        output, _ = self.decoder_model(output)
        output = self.dropout_lstm(output)
        output = self.decoder_dense_output(torch.tanh(output[:, -1]))

        return output
