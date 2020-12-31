'''
A 3-layer Convolutional Neural Network.
'''

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from meta_classes import ModelMeta

class SimpleCNN(nn.Module, ModelMeta):
    '''
    3-Layer CNN.
    '''

    def __init__(self, image_size, image_depth, num_classes, drop_prob, decay_rate, learning_rate, device):
        '''
        Initialize the parameters.
        '''
        super(SimpleCNN, self).__init__()

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

        self.simple_cnn = nn.Sequential(nn.Conv2d(in_channels=self.image_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_vector_size = (self.image_size//(2**3))**2 * 256

        self.fully_connected = nn.Sequential(nn.Linear(self.feature_vector_size, 128),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(p=self.drop_prob),
                                             nn.Linear(128, self.num_classes))




    @staticmethod
    def loss_optim_init(model_obj, decay_rate):
        '''
        Initialize the loss function and the optimizer for the given object model.
        '''
        loss_func = nn.CrossEntropyLoss()
        optim_func = torch.optim.Adam(model_obj.parameters())
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optim_func, gamma=decay_rate)

        return loss_func, optim_func, lr_decay



    def forward(self, x):
        '''
        Forward-propagation
        '''

        x = self.simple_cnn(x)
        x = torch.flatten(x, 1)
        output = self.fully_connected(x)

        return output


    @staticmethod
    def optimize_model(predicted, target, loss_func, optim_func, decay_func, lr_decay_step=False):
        '''
        Model optimization.
        '''

        optim_func.zero_grad()
        total_loss = loss_func(input=predicted, target=target)
        total_loss.backward()

        optim_func.step()

        if lr_decay_step:
            decay_func.step()

        return total_loss.item()


    def calculate_loss(predicted, target, loss_func):
        '''
        Returns the loss of the model without optimizing the model.
        '''

        total_loss = loss_func(input=predicted, target=target)

        return total_loss.item()

    @staticmethod
    def calculate_accuracy(predicted, target):
        '''
        Calculates the accuracy of the predictions.
        '''

        num_data = target.size()[0]
        predicted = torch.argmax(predicted, dim=1)
        correct_pred = torch.sum(predicted == target)

        accuracy = correct_pred*(100/num_data)

        return accuracy.item()
