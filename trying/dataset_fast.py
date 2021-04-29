import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models

import numpy as np
import os
import sys
import pickle

from keras.preprocessing.image import img_to_array, load_img

from utils import *

class DatasetTrain(torch.utils.data.Dataset):

    def __init__(self, train, len):

        self.input, self.label = train
        self.input1 = self.input[0]
        self.input2 = self.input[1]
        self.num_examples = len


    def __getitem__(self, index):
        return torch.from_numpy(self.input1[index]), torch.from_numpy(self.input2[index]), \
               torch.from_numpy(self.label[index])


    def __len__(self):
        return self.num_examples


class DatasetVal(torch.utils.data.Dataset):

    def __init__(self, val, len):

        self.input, self.label = val
        self.num_examples = len

    def __getitem__(self, index):

        return torch.from_numpy(self.input[0][index]), torch.from_numpy(self.input[1][index]),\
               torch.from_numpy(self.label[index])

    def __len__(self):
        return self.num_examples