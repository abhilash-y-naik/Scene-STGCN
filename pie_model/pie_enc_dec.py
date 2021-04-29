import torch
import torch.nn as nn

from pie_model.convlstm2D import ConvLSTM2D

import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable

class pie_convlstm_encdec(nn.Module):
    '''
    Create an LSTM Encoder-Decoder model for intention estimation
    '''

    def __init__(self, learning_rate_params,
                 num_hidden_units=128,
                 decoder_seq_length=15,
                 lstm_dropout=0.4,
                 convlstm_num_filters=64,
                 convlstm_kernel_size=2,
                 ):
        super(pie_convlstm_encdec, self).__init__()

        # Learning rate params
        self.lr_params = learning_rate_params

        # Network parameters
        self._num_hidden_units = num_hidden_units

        # Encoder
        self._lstm_dropout = int(lstm_dropout)

        # conv unit parameters
        self._convlstm_num_filters = convlstm_num_filters
        self._convlstm_kernel_size = convlstm_kernel_size

        # decoder unit parameters
        self._decoder_dense_output_size = 1
        self._decoder_input_size = 4  # decided on run time according to data
        self._decoder_seq_length = decoder_seq_length

        # Generate Encoder LSTM Unit
        self.encoder_model = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False)

        self.conv = nn.Conv2d(self._convlstm_num_filters, self._convlstm_num_filters,
                              kernel_size=self._convlstm_kernel_size, bias=True)
        nn.init.xavier_normal_(self.conv.weight)
        self.batch = nn.BatchNorm2d(self._convlstm_num_filters)
        self.dropout = nn.Dropout(p=self._lstm_dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.encoder_output = nn.Flatten()

        # Generate Decoder LSTM unit
        self.decoder_model = nn.LSTM(2308, self._num_hidden_units,
                                     bias=True, batch_first=True)
        nn.init.xavier_normal_(self.decoder_model.weight_hh_l0)
        self.dropout_lstm = nn.Dropout(p=self._lstm_dropout)

        self.decoder_dense_output = nn.Linear(self._num_hidden_units, self._decoder_dense_output_size, bias=True)
        nn.init.xavier_normal_(self.decoder_dense_output.weight)

    def forward(self, encoder_input, decoder_input):

        B, T, H, W, C = encoder_input.size()
        output = encoder_input.view((B, T, C, H, W))
        _, output = self.encoder_model(output)
        output = self.pool(output)
        output = self.encoder_output(output)
        repeat_vec = output.repeat(1, self._decoder_seq_length).reshape(B, self._decoder_seq_length, -1)
        output = torch.cat((repeat_vec, decoder_input), dim=2)
        output, _ = self.decoder_model(output)
        output = self.dropout_lstm(output)
        output = self.decoder_dense_output(torch.tanh(output[:, -1]))

        return output

# To check parameters using pytorch lightning module
# class pie_convlstm_encdec(pl.LightningModule):
#     '''
#     Create an LSTM Encoder-Decoder model for intention estimation
#     '''
#
#     def __init__(self, learning_rate_params,
#                  num_hidden_units=128,
#                  decoder_seq_length=15,
#                  lstm_dropout=0.4,
#                  convlstm_num_filters=64,
#                  convlstm_kernel_size=2,
#                  ):
#         super(pie_convlstm_encdec, self).__init__()
#
#         # Learning rate params
#         self.lr_params = learning_rate_params
#
#
#         # Network parameters
#         self._num_hidden_units = num_hidden_units
#
#         # Encoder
#         self._lstm_dropout = int(lstm_dropout)
#
#         # conv unit parameters
#         self._convlstm_num_filters = convlstm_num_filters
#         self._convlstm_kernel_size = convlstm_kernel_size
#
#         # decoder unit parameters
#         self._decoder_dense_output_size = 1
#         self._decoder_input_size = 4  # decided on run time according to data
#         self._decoder_seq_length = decoder_seq_length
#
#         # Generate Encoder LSTM Unit
#         self.encoder_model = ConvLSTM2D(512, self._convlstm_num_filters, dropout=self._lstm_dropout, kernel_size=(2, 2), num_layers=1,
#                  batch_first=True, bias=True, return_all_layers=False)
#
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.encoder_output = nn.Flatten()
#
#         # Generate Decoder LSTM unit
#         self.decoder_model = nn.LSTM(2308, self._num_hidden_units,
#                                      bias=True, batch_first=True)
#         nn.init.xavier_normal_(self.decoder_model.weight_hh_l0)
#         self.dropout_lstm = nn.Dropout(p=self._lstm_dropout)
#
#         self.decoder_dense_output = nn.Linear(self._num_hidden_units, self._decoder_dense_output_size, bias=True)
#         nn.init.xavier_normal_(self.decoder_dense_output.weight)
#
#     def forward(self, encoder_input, decoder_input):
#
#         B, T, H, W,C = encoder_input.size()
#         output = encoder_input.view((B, T, C, H, W))
#         _, output = self.encoder_model(output)
#         output = self.pool(output)
#
#         # output = F.relu(self.batch(self.conv(output)))
#
#         output = self.encoder_output(output)
#         repeat_vec = output.repeat(1, self._decoder_seq_length).reshape(B, self._decoder_seq_length, -1)
#         output = torch.cat((repeat_vec, decoder_input), dim=2)
#         output, _ = self.decoder_model(output)
#         output = self.dropout_lstm(output)
#         output = self.decoder_dense_output(torch.tanh(output[:, -1]))
#         # output = torch.sigmoid(output)
#
#         return output
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
#
#     def training_step(self, train_batch, batch_idx):
#         input_enc, input_dec, label = train_batch
#         input_enc = Variable(input_enc).cuda()
#         input_dec = Variable(input_dec.type(torch.FloatTensor)).cuda()
#         label = Variable(label.type(torch.float)).cuda()
#         z = pie_convlstm_encdec(input_enc,input_dec)
#         loss = F.mse_loss(z, label)
#         self.log('train_loss', loss)
#         return loss