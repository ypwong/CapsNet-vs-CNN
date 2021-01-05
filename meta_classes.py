'''
Interface and abstract classes.
'''

from abc import ABCMeta, abstractmethod


class DataMeta(metaclass=ABCMeta):
    '''
    Interface class for the datasets.
    '''

    @abstractmethod
    def get_feature_level_data(self):
        '''
        Return the names of the feature-level classes and their corresponding index value in a dictionary.
        '''
        pass

    @abstractmethod
    def get_object_level_data(self):
        '''
        Return the name of the object-level classes and their corresponding index value in a dictionary.
        '''

    @abstractmethod
    def gen_feature_level(self):
        '''
        Generates a single feature-level data image from each feature-level classes.
        '''
        pass

    @abstractmethod
    def gen_object_level(self):
        '''
        Generates a single object-level data image from each object-level classes.
        '''
        pass


class ModelMeta(metaclass=ABCMeta):
    '''
    Interface class for the machine learning models.
    '''

    @abstractmethod
    def build_model(self):
        '''
        Builds the architecture of the model.
        '''
        pass


    @abstractmethod
    def optimize_model():
        '''
        Returns the loss after optimizing the given model using the loss, optimzer and lr decayer given in the parameter.
        '''
        pass


    @abstractmethod
    def loss_optim_init():
        '''
        Init loss, optim and lr decayer functions for the model and returns them to be used by the client.
        '''
        pass


    @abstractmethod
    def calculate_loss():
        '''
        Returns the loss of the network without optimizing the model.
        '''
        pass

    @staticmethod
    @abstractmethod
    def calculate_accuracy():
        '''
        Calculates the accuracy on a given dataset.
        '''
        pass
