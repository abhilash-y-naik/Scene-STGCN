"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import pytorch_lightning as pl  # Faster way to verify model parameters

import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from  sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from pie_model.utils import *
from pie_model.dataset import DatasetTrain, DatasetVal
from pie_model.pie_enc_dec import pie_convlstm_encdec


class PIEIntent(object):
    """
    A convLSTM encoder decoder model for predicting pedestrian intention
    Attributes:
        _num_hidden_units: Number of LSTM hidden units
        _reg_value: the value of L2 regularizer for training
        _kernel_regularizer: Training regularizer set as L2
        _recurrent_regularizer: Training regularizer set as L2
        _activation: LSTM activations
        _lstm_dropout: input dropout
        _lstm_recurrent_dropout: recurrent dropout
        _convlstm_num_filters: number of filters in convLSTM
        _convlstm_kernel_size: kernel size in convLSTM

    Model attributes: set during training depending on the data
        _encoder_input_size: size of the encoder input
        _decoder_input_size: size of the encoder_output

    Methods:
        load_images_and_process: generates trajectories by sampling from pedestrian sequences
        get_data_slices: generate tracks for training/testing
        create_lstm_model: a helper function for creating conv LSTM unit
        pie_convlstm_encdec: generates intention prediction model
        train: trains the model
        test_chunk: tests the model (chunks the test cases for memory efficiency)
    """

    def __init__(self,
                 num_hidden_units=128,
                 regularizer_val=0.001,
                 activation='tanh',
                 lstm_dropout=0.4,
                 lstm_recurrent_dropout=0.2,
                 convlstm_num_filters=64,
                 convlstm_kernel_size=2):

        # Network parameters
        self._num_hidden_units = num_hidden_units
        self.reg_value = regularizer_val
        self._kernel_regularizer = regularizer_val
        self._recurrent_regularizer = regularizer_val
        self._bias_regularizer = regularizer_val
        self._activation = activation

        # Encoder
        self._lstm_dropout = lstm_dropout
        self._lstm_recurrent_dropout = lstm_recurrent_dropout

        # conv unit parameters
        self._convlstm_num_filters = convlstm_num_filters
        self._convlstm_kernel_size = convlstm_kernel_size

        # self._encoder_dense_output_size = 1 # set this only for single lstm unit
        self._encoder_input_size = 4  # decided on run time according to data

        self._decoder_dense_output_size = 1
        self._decoder_input_size = 4  # decided on run time according to data

        # Data properties
        # self._batch_size = 128  # this will be set at train time
        self._model_name = 'convlstm_encdec'


    def get_model_config(self):
        """
        Returns a dictionary containing model configuration.
        """

        config = dict()
        # Network parameters
        config['num_hidden'] = self._num_hidden_units
        # config['bias_init'] = self._bias_initializer
        config['reg_value'] = self.reg_value
        config['activation'] = self._activation
        config['sequence_length'] = self._sequence_length
        config['lstm_dropout'] = self._lstm_dropout
        config['lstm_recurrent_dropout'] = self._lstm_recurrent_dropout
        config['convlstm_num_filters'] = self._convlstm_num_filters
        config['convlstm_kernel_size'] = self._convlstm_kernel_size

        config['encoder_input_size'] = self._encoder_input_size

        config['decoder_input_size'] = self._decoder_input_size
        config['decoder_dense_output_size'] = self._decoder_dense_output_size

        # Set the input sizes
        config['encoder_seq_length'] = self._encoder_seq_length
        config['decoder_seq_length'] = self._decoder_seq_length

        print(config)
        return config

    def load_model_config(self, config):
        """
        Copy config information from the dictionary for testing
        """
        # Network parameters
        self._num_hidden_units = config['num_hidden']

        self.reg_value = config['reg_value']
        self._activation = config['activation']
        self._encoder_input_size = config['encoder_input_size']
        self._encoder_seq_length = config['encoder_seq_length']
        self._sequence_length = config['sequence_length']

        self._lstm_dropout = config['lstm_dropout']
        self._lstm_recurrent_dropout = config['lstm_recurrent_dropout']
        self._convlstm_num_filters = config['convlstm_num_filters']
        self._convlstm_kernel_size = config['convlstm_kernel_size']

        self._encoder_input_size = config['decoder_input_size']
        self._decoder_input_size = config['decoder_input_size']
        self._decoder_dense_output_size = config['decoder_dense_output_size']
        self._decoder_seq_length = config['decoder_seq_length']

    def get_path(self,
                 type_save='models',  # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder='./data/'):

        """
        A path generator method for saving model and config data. Creates directories
        as needed.
        :param type_save: Specifies whether data or model is saved.
        :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
        :param model_name: model name (either trained convlstm_encdec model or vgg16)
        :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
        :param data_subset: train, test or val
        :param data_type: type of the data (e.g. features_context_pad_resize)
        :param save_root_folder: The root folder for saved data.
        :return: The full path for the save folder
        """
        assert (type_save in ['models', 'data'])
        if data_type != '':
            assert (any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def get_tracks(self, dataset, data_type, seq_length, overlap):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        :param data_type: types of data for encoder/decoder input
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: a dictionary containing sampled tracks for each data modality
        """
        overlap_stride = seq_length if overlap == 0 else \
            int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d_types = []
        for k in data_type.keys():
            d_types.extend(data_type[k])
        d = {}

        if 'bbox' in d_types:
            d['bbox'] = dataset['bbox']
        if 'intention_binary' in d_types:
            d['intention_binary'] = dataset['intention_binary']
        if 'intention_prob' in d_types:
            d['intention_prob'] = dataset['intention_prob']

        bboxes = dataset['bbox'].copy()
        images = dataset['image'].copy()
        ped_ids = dataset['ped_id'].copy()

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in \
                               range(0, len(track) \
                                     - seq_length + 1, overlap_stride)])
            d[k] = tracks

        pid = []
        for p in ped_ids:
            pid.extend([p[i:i + seq_length] for i in \
                        range(0, len(p) \
                              - seq_length + 1, overlap_stride)])
        ped_ids = pid

        im = []
        for img in images:
            im.extend([img[i:i + seq_length] for i in \
                       range(0, len(img) \
                             - seq_length + 1, overlap_stride)])
        images = im

        bb = []
        for bbox in bboxes:
            bb.extend([bbox[i:i + seq_length] for i in \
                       range(0, len(bbox) \
                             - seq_length + 1, overlap_stride)])

        bboxes = bb

        return d, images, bboxes, ped_ids

    def concat_data(self, data, data_type):
        """
        Concatenates different types of data specified by data_type.
        Creats dummy data if no data type is specified
        :param data_type: type of data (e.g. bbox)
        """
        if not data_type:
            return []
        # if more than one data type is specified, they are concatenated
        d = []
        for dt in data_type:
            d.append(np.array(data[dt]))
        if len(d) > 1:
            d = np.concatenate(d, axis=2)
        else:
            d = d[0]
        return d

    def get_train_val_data(self, data, data_type, seq_length, overlap):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """
        tracks, images, bboxes, ped_ids = self.get_tracks(data, data_type, seq_length, overlap)

        # Generate observation data input to encoder
        encoder_input = self.concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = self.concat_data(tracks, data_type['decoder_input_type'])
        output = self.concat_data(tracks, data_type['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)

        return {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'output': output}

    def get_model(self, model):

        train_config = {'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                      'step_drop_rate': 0.5,
                                                      'epochs_drop_rate': 20.0,
                                                      'plateau_patience': 5,
                                                      'min_lr': 0.0000001,
                                                      'monitor_value': 'val_loss'}}

        train_model = pie_convlstm_encdec(train_config['learning_scheduler_params'],
                                          num_hidden_units=self._num_hidden_units,
                                         decoder_seq_length=self._decoder_seq_length,
                                         lstm_dropout=self._lstm_dropout,
                                         convlstm_num_filters=self._convlstm_num_filters,
                                         convlstm_kernel_size=self._convlstm_kernel_size,
                                         ).cuda()
        return train_model


    def train(self,
              data_train,
              data_val,
              batch_size=128,
              epochs=400,
              optimizer_type='rmsprop',
              optimizer_params={'lr': 0.00001, 'clipvalue': 0.0, 'decay': 0.0},
              loss=['binary_crossentropy'],
              metrics=['acc'],
              data_opts=''):

        """
        Training method for the model
        :param data_train: training data
        :param data_val: validation data
        :param batch_size: batch size for training
        :param epochs: number of epochs for training
        :param optimizer_params: learning rate and clipvalue for gradient clipping
        :param loss: type of loss function
        :param metrics: metrics to monitor
        :param data_opts: data generation parameters
        """
        data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                     'decoder_input_type': data_opts['decoder_input_type'],
                     'output_type': data_opts['output_type']}

        train_config = {'batch_size': batch_size,
                        'epoch': epochs,
                        'optimizer_type': optimizer_type,
                        'optimizer_params': optimizer_params,
                        'loss': loss,
                        'metrics': metrics,
                        'learning_scheduler_mode': 'plateau',
                        'seed': 170349028355500,
                        'lambda_l2': 0.05,
                        'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                      'step_drop_rate': 0.5,
                                                      'epochs_drop_rate': 20.0,
                                                      'plateau_patience': 5,
                                                      'min_lr': 0.0000001,
                                                      'monitor_value': 'val_loss'},
                        'model': 'convlstm_encdec',
                        'data_type': data_type,
                        'overlap': data_opts['seq_overlap_rate'],
                        'dataset': 'pie'}

        self._model_type='convlstm_encdec'

        seq_length = data_opts['max_size_observe']
        train_d = self.get_train_val_data(data_train, data_type, seq_length, data_opts['seq_overlap_rate'])
        val_d = self.get_train_val_data(data_val, data_type, seq_length, data_opts['seq_overlap_rate'])

        self._encoder_seq_length = train_d['decoder_input'].shape[1]
        self._decoder_seq_length = train_d['decoder_input'].shape[1]

        self._sequence_length = self._encoder_seq_length

        train_dataset = DatasetTrain(train_d, data_opts)
        val_dataset = DatasetVal(val_d, train_d, data_opts)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=8
                                                   )

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=batch_size, shuffle=False, num_workers=8
                                                 )


        # print(torch.seed())
        # automatically generate model name as a time string
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        model_path, _ = self.get_path(type_save='models',
                                      model_name='convlstm_encdec',
                                      models_save_folder=model_folder_name,
                                      file_name='model.pth',
                                      save_root_folder='data')
        config_path, _ = self.get_path(type_save='models',
                                       model_name='convlstm_encdec',
                                       models_save_folder=model_folder_name,
                                       file_name='configs',
                                       save_root_folder='data')

        # Save config and training param files
        with open(config_path + '.pkl', 'wb') as fid:
            pickle.dump([self.get_model_config(),
                         train_config, data_opts],
                        fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        # Save config and training param files
        with open(config_path + '.txt', 'wt') as fid:
            fid.write("####### Data options #######\n")
            fid.write(str(data_opts))
            fid.write("\n####### Model config #######\n")
            fid.write(str(self.get_model_config()))
            fid.write("\n####### Training config #######\n")
            fid.write(str(train_config))


        train_model = pie_convlstm_encdec(train_config['learning_scheduler_params'],
                                          num_hidden_units=self._num_hidden_units,
                                         decoder_seq_length=self._decoder_seq_length,
                                         lstm_dropout=self._lstm_dropout,
                                         convlstm_num_filters=self._convlstm_num_filters,
                                         convlstm_kernel_size=self._convlstm_kernel_size,
                                         ).cuda()


        ## To check paramters using pytorch lightning

        # trainer = pl.Trainer()
        # trainer.fit(train_model, train_dataloader=train_loader)

        #################################################################################################
        # Optimisers , learning rate schedulers and loss function
        optimizer = torch.optim.RMSprop(train_model.parameters(),
                                    lr=optimizer_params['lr'],
                                    weight_decay=optimizer_params['decay'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=train_config['learning_scheduler_params']['step_drop_rate'],
                                                        patience=int(train_config['learning_scheduler_params']['plateau_patience']),
                                                        min_lr=train_config['learning_scheduler_params']['min_lr'],
                                                        verbose=True)

        loss_fn = nn.BCEWithLogitsLoss()

        #################################################################################################

        epoch_losses_train = []
        epoch_accuracy_train = []

        epoch_losses_val = []
        epoch_accuracy_val = []
        for epoch in range(epochs):
            accuracy = 0
            count = 0
            print("###########################")
            print("######## NEW EPOCH ########")
            print("###########################")
            print("epoch: %d/%d" % (epoch + 1, epochs))

            ###########################################################################
            # train:
            ###########################################################################
            train_model.train()  # (set in training mode, this affects BatchNorm and dropout)
            batch_losses = []
            y_true = []
            y_pred = []
            for step, (input_enc, input_dec, label) in enumerate(train_loader):
                if count % 10 == 0:
                    print(count)
                count = count + 1

                input_enc = Variable(input_enc).cuda()
                input_dec = Variable(input_dec.type(torch.FloatTensor)).cuda()
                label = Variable(label.type(torch.float)).cuda()

                outputs = train_model(input_enc, input_dec)

                loss = loss_fn(outputs, label)
                l2_enc = torch.cat([x.view(-1) for x in train_model.encoder_model.parameters()])
                l2_dec = torch.cat([x.view(-1) for x in train_model.decoder_model.parameters()])

                loss = loss + train_config['lambda_l2']*torch.norm(l2_enc, p=2) + \
                       train_config['lambda_l2']*torch.norm(l2_dec, p=2)

                y_true.append(np.asarray(label.data.to('cpu')))
                y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))

                batch_losses.append(loss.data.cpu().numpy())

                # optimization step:
                optimizer.zero_grad()  # (reset gradients)
                loss.backward()  # (compute gradients)
                optimizer.step()  # (perform optimization step)

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            accuracy = accuracy_score(y_true, y_pred)
            epoch_loss = np.mean(batch_losses)
            epoch_losses_train.append(epoch_loss)
            epoch_accuracy_train.append(accuracy)
            print("train loss: %g" % epoch_loss)
            print("Accuracy:  %g" % accuracy)
            print("####")

            # #
            # # ############################################################################
            # # # val:
            # # ############################################################################
            train_model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
            batch_losses = []
            y_true = []
            y_pred = []
            val_loss = 0
            count = 0
            for step, (input_enc, input_dec, label) in enumerate(val_loader):
                with torch.no_grad():

                    if count % 10 == 0:
                        print(count)
                    count = count + 1

                    input_enc = Variable(input_enc).cuda()
                    input_dec = Variable(input_dec.type(torch.FloatTensor)).cuda()
                    label = Variable(label.type(torch.float)).cuda()

                    outputs = train_model(input_enc, input_dec)
                    loss = loss_fn(outputs, label)

                    y_true.append(np.asarray(label.data.to('cpu')))
                    y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))
                    batch_losses.append(loss.data.cpu().numpy())
                    val_loss += loss

            scheduler.step(val_loss / count)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            accuracy = accuracy_score(y_true, y_pred)

            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            epoch_accuracy_val.append(accuracy)

            print("val loss: %g" % epoch_loss)
            print("Accuracy:  %g" % accuracy)

            plt.figure(1)
            plt.plot(epoch_losses_train, "r^")
            plt.plot(epoch_losses_train, "r", label='train')
            plt.plot(epoch_losses_val, "k^")
            plt.plot(epoch_losses_val, "k", label='val')
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.title("loss per epoch")
            plt.legend()
            plt.savefig("%s/epoch_losses.png" % model_path.split("model.pth")[0])
            plt.close(1)

            plt.figure(2)
            plt.plot(epoch_accuracy_train, "r^")
            plt.plot(epoch_accuracy_train, "r", label='train')
            plt.plot(epoch_accuracy_val, "k^")
            plt.plot(epoch_accuracy_val, "k", label='val')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.title("accuracy per epoch")
            plt.legend()
            plt.savefig("%s/epoch_accuracy.png" % model_path.split("model.pth")[0])
            plt.close(2)

            if (epoch+1) % 50 == 0:
                model_saving_path = model_path.split("model.pth")[0] + "/model_" + "epoch_" + str(epoch+1) + ".pth"
                torch.save(train_model.state_dict(), model_saving_path)



#############################################################################################
# Testing code
#############################################################################################

    class DatasetTest(torch.utils.data.Dataset):

        def __init__(self, images, bboxes, ped_ids, decoder_input, output):

            """
            A helper function for test data generation that preprocesses the images, combines
            different representations required as inputs to encoder and decoder, as well as
            ground truth and returns them as a unified list representation.
            :param data: A dictionary of data types
            :param train_params: Training parameters defining the type of
            :param data_type: The data types defined for encoder and decoder
            :return: A unified data representation as a list.
            """

            self.images = images
            self.bboxes = bboxes
            self.ped_ids = ped_ids
            self.decoder_input = decoder_input
            self.output = output

            # Create context model
            vgg16 = models.vgg16_bn()
            vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
            self.context_model = nn.Sequential(*list(vgg16.children())[:-1])

            self.num_examples = len(self.ped_ids)
        def __getitem__(self, index):

            test_img = self.load_images_and_process(self.images[index],
                                                    self.bboxes[index],
                                                    self.ped_ids[index],
                                                    data_type='test',
                                                    save_path=self.get_path(type_save='data',
                                                                        data_type='features_context_pad_resize',
                                                                        # images
                                                                        model_name='vgg16_none',
                                                                        data_subset='test'))


            output = self.output[index][0]
            return torch.from_numpy(test_img), torch.from_numpy(self.decoder_input[index]), torch.from_numpy(output)

        def get_path(self,
                     type_save='models',  # model or data
                     models_save_folder='',
                     model_name='convlstm_encdec',
                     file_name='',
                     data_subset='',
                     data_type='',
                     save_root_folder='./data/'):

            """
            A path generator method for saving model and config data. Creates directories
            as needed.
            :param type_save: Specifies whether data or model is saved.
            :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
            :param model_name: model name (either trained convlstm_encdec model or vgg16)
            :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
            :param data_subset: train, test or val
            :param data_type: type of the data (e.g. features_context_pad_resize)
            :param save_root_folder: The root folder for saved data.
            :return: The full path for the save folder
            """
            assert (type_save in ['models', 'data'])
            if data_type != '':
                assert (any([d in data_type for d in ['images', 'features']]))
            root = os.path.join(save_root_folder, type_save)

            if type_save == 'models':
                save_path = os.path.join(save_root_folder, 'pie', 'intention', models_save_folder)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                return os.path.join(save_path, file_name), save_path
            else:
                save_path = os.path.join(root, 'pie', data_subset, data_type, model_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                return save_path

        def load_images_and_process(self,
                                    img_sequences,
                                    bbox_sequences,
                                    ped_ids,
                                    save_path,
                                    data_type='train',
                                    regen_pkl=False):
            """
            Generates image features for convLSTM input. The images are first
            cropped to 1.5x the size of the bounding box, padded and resized to
            (224, 224) and fed into pretrained VGG16.
            :param img_sequences: a list of frame names
            :param bbox_sequences: a list of corresponding bounding boxes
            :ped_ids: a list of pedestrian ids associated with the sequences
            :save_path: path to save the precomputed features
            :data_type: train/val/test data set
            :regen_pkl: if set to True overwrites previously saved features
            :return: a list of image features
            """
            try:
                convnet = self.context_model
            except:
                raise Exception("No context model is defined")

            img_seq = []
            for imp, b, p in zip(img_sequences, bbox_sequences, ped_ids):

                imp = imp.replace(os.sep, '/')
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')
                if os.path.exists(img_save_path) and not regen_pkl:
                    with open(img_save_path, 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                else:
                    img_data = load_img(imp)
                    bbox = jitter_bbox(imp, [b], 'enlarge', 2)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    bbox = list(map(int, bbox[0:4]))
                    cropped_image = img_data.crop(bbox)
                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                    image_array = img_to_array(img_data).reshape(-1, 224, 224)
                    image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float()
                    img_features = convnet(image_array)
                    B, C, H, W = img_features.size()
                    img_features = img_features.view((B, H, W, C))
                    img_features = img_features.data.to('cpu').numpy()
                    print(img_save_path)
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)
                img_features = np.squeeze(img_features)
                img_seq.append(img_features)
            sequences = np.array(img_seq)

            return sequences

        def __len__(self):
            return self.num_examples

    # split test data into chunks
    def test_chunk(self,
                   data_test,
                   data_opts='',
                   model_path='',
                   visualize=False):

        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except:
                configs = pickle.load(fid, encoding='bytes')
        train_params = configs[1]
        self.load_model_config(configs[0])

        test_model = self.get_model(train_params['model'])
        test_model.load_state_dict(torch.load(os.path.join(model_path, 'model_epoch_50.pth')))
        print('epoch:50')

        overlap = 1  # train_params ['overlap']

        tracks, images, bboxes, ped_ids = self.get_tracks(data_test,
                                                          train_params['data_type'],
                                                          self._sequence_length,
                                                          overlap)

        # Generate observation data input to encoder
        decoder_input = self.concat_data(tracks, train_params['data_type']['decoder_input_type'])
        output = self.concat_data(tracks, train_params['data_type']['output_type'])
        if len(decoder_input) == 0:
           decoder_input = np.zeros(shape=np.array(bboxes).shape)

        test_dataset = self.DatasetTest(images, bboxes, ped_ids, decoder_input,
                                        output)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=128, shuffle=False, num_workers=8
                                                  )

        #####################################################################################

        test_model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
        count = 0
        y_true = []
        y_pred = []
        new_pred = []
        for step, (input_enc, input_dec, label) in enumerate(test_loader):
            with torch.no_grad():
                    if count % 10 == 0:
                        print(count)
                    count = count + 1

                    input_enc = Variable(input_enc).cuda()
                    input_dec = Variable(input_dec.type(torch.FloatTensor)).cuda()
                    label = Variable(label.type(torch.float)).cuda()

                    outputs = test_model(input_enc, input_dec)

                    y_true.append(np.asarray(label.data.to('cpu')))
                    y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))
                    # new_pred.append(torch.sigmoid(outputs).data.to('cpu'))

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        # new_pred = np.concatenate(new_pred, axis=0)
        TN = confusion_matrix(y_true, y_pred)[0, 0]
        FP = confusion_matrix(y_true, y_pred)[0, 1]
        FN = confusion_matrix(y_true, y_pred)[1, 0]
        TP = confusion_matrix(y_true, y_pred)[1, 1]
        print('CONFUSION MATRIX:')
        print("TP: %g" % TP, "FP: %g" % FP)
        print("FN: %g" % FN, "TN: %g" % TN)

        # fpr, tpr, thresholds = roc_curve(y_true, new_pred)

        # gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        # ix = np.argmax(gmeans)
        #
        # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        # # plot the roc curve for the model
        # pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # pyplot.plot(fpr, tpr, marker='.', label='Logistic')
        # pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # # axis labels
        # pyplot.xlabel('False Positive Rate')
        # pyplot.ylabel('True Positive Rate')
        # pyplot.legend()
        # # show the plot
        # pyplot.show()

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        acc = accuracy_score(y_true, y_pred)
        return acc, f1, precision, recall



