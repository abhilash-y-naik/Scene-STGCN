import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable

import numpy as np
import os
import pickle

from pie_model.utils import *

class DatasetTrain(torch.utils.data.Dataset):

    def __init__(self, train, data_opts):

        self.data_opts = data_opts
        self.train_d = train

        # Create context model
        vgg16 = models.vgg16_bn().cuda()
        vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
        self.context_model = nn.Sequential(*list(vgg16.children())[:-1])

        self.num_examples = len(self.train_d['ped_ids'])

    def __getitem__(self, index):

        # crop only bounding boxes
        train_img = self.load_images_and_process(self.train_d['images'][index],
                                                 self.train_d['bboxes'][index],
                                                 self.train_d['ped_ids'][index],
                                                 data_type='train',
                                                 save_path=self.get_path(type_save='data',
                                                                         data_type='features' + '_' + self.data_opts[
                                                                             'crop_type'] + '_' + self.data_opts[
                                                                                       'crop_mode'],  # images
                                                                         model_name='vgg16_' + 'none',
                                                                         data_subset='train'))

        train_data = torch.from_numpy(train_img), torch.from_numpy(self.train_d['decoder_input'][index]), torch.from_numpy(self.train_d['output'][index][0])
        return train_data

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
                image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                img_features = convnet(image_array)
                B, C, H, W = img_features.size()
                img_features = img_features.view((B, H, W, C))
                img_features = img_features.data.to('cpu').numpy()
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


class DatasetVal(torch.utils.data.Dataset):

    def __init__(self, val, train, data_opts):

        self.data_opts = data_opts
        self.val_d = val
        self.train_d = train

        # Create context model
        vgg16 = models.vgg16_bn().cuda()
        vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
        self.context_model = nn.Sequential(*list(vgg16.children())[:-1])

        self.num_examples = len(self.val_d['ped_ids'])


    def __getitem__(self, index):

        # crop only bounding boxes
        val_img = self.load_images_and_process(self.val_d['images'][index],
                                                 self.val_d['bboxes'][index],
                                                 self.train_d['ped_ids'][index],
                                                 data_type='val',
                                                 save_path=self.get_path(type_save='data',
                                                                         data_type='features' + '_' + self.data_opts[
                                                                             'crop_type'] + '_' + self.data_opts[
                                                                                       'crop_mode'],  # images
                                                                         model_name='vgg16_' + 'none',
                                                                         data_subset='val'))

        val_data = torch.from_numpy(val_img), torch.from_numpy(self.val_d['decoder_input'][index]), torch.from_numpy(self.val_d['output'][index][0])
        return val_data

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
                                data_type='val',
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
        # load the feature files if exists
        # print("Generating {} features crop_type=context crop_mode=pad_resize \nsave_path={}, ".format(data_type,
        #                                                                                               save_path))
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
                image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                img_features = convnet(image_array)
                B, C, H, W = img_features.size()
                img_features = img_features.view((B, H, W, C))
                img_features = img_features.data.to('cpu').numpy()
                # print(img_save_path)
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