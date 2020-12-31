'''
A 6-layer Convolutional Neural Network.
'''

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from meta_classes import ModelMeta


class DeepCNN(nn.Module, ModelMeta):
    '''
    6-Layer CNN.
    '''

    def __init__(self, image_size, image_depth, num_classes, drop_prob, decay_rate, learning_rate, device):
        '''
        Param init.
        '''

        super(DeepCNN, self).__init__()

        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.device = device


    def build_model(self):
        '''
        Build architecture of the model.
        '''

        self.deep_cnn = nn.Sequential(nn.Conv2d(in_channels=self.image_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stide=2),
                                      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      )


