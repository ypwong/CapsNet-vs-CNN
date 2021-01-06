'''
Factory that initializes and provides a model as a Python Object.
'''
import sys
import torch
from .simple_cnn import SimpleCNN
from .simple_capsnet import SimpleCapsNet
from .simple_cnn_conf import defaults as simple_cnn_defaults
from .simple_capsnet_conf import defaults as simple_capsnet_defaults


class ModelFactory:

    def factory_call(self, requested_model, mode=None, model_load_path=None, **kwargs):
        '''
        Returns the implementation of the requested model and its optimizing function, loss function and lr decayer function as python objects.
        '''

        assert mode is None or mode=='transfer_learning', "Mode must be either None (feature-level) or 'transfer_learning' (object-level)"

        ########################################### SIMPLE CNN ###########################################
        if requested_model == 'simple_cnn':

            parameters = simple_cnn_defaults

            for key, value in kwargs.items():
                try:
                    parameters[key] = value
                except KeyError:
                    raise KeyError("The parameter that you have supplied is not valid for this dataset!")

            model_object = SimpleCNN(**parameters)
            model_object.build_model()
            loss_func, optim_func, lr_decay_func = model_object.loss_optim_init(model_object, decay_rate=parameters['decay_rate'])

            if mode == 'transfer_learning':
                #if mode is transfer-learning it implies the training is for the object-level dataset. Therefore, the weights and biases in the
                #convolutional layers will be freezed to retain the learned features.

                assert not model_load_path is None, "The trained model path must be provided for transfer-learning mode!"
                model_load_path = model_load_path.rstrip('/') + '/simple_cnn_feature_model.pth'
                try:
                    model_object.load_state_dict(torch.load(model_load_path))
                    print("Model has loaded!")

                    #disables the gradient flow in the convolutional layers.
                    for x in model_object.simple_cnn.parameters():
                        x.requires_grad = False
                    print("The gradient flow in the convolutional layers has been disabled!")

                except Exception as e:
                    print("Error loading the trained model!")
                    print(e)
                    print("--- Exiting ---")
                    sys.exit()

            return model_object, loss_func, optim_func, lr_decay_func

        ########################################### SIMPLE CAPSNET ###########################################
        elif requested_model == 'simple_capsnet':

            parameters = simple_capsnet_defaults

            for key, value in kwargs.items():
                try:
                    parameters[key] = value
                except KeyError:
                    raise KeyError("The parameter that you have supplied is not valid for this dataset!")

            model_object = SimpleCapsNet(**parameters)
            model_object.build_model()
            loss_func, optim_func, lr_decay_func = model_object.loss_optim_init(model_object)

            if mode == 'transfer_learning':

                assert not model_load_path is None, "The trained model path must be provided for transfer-learning mode!"
                model_load_path = model_load_path.rstrip('/') + '/simple_capsnet_feature_model.pth'
                try:
                    model_object.load_state_dict(torch.load(model_load_path))
                    print("Model has loaded!")

                    #disables the gradient flow in the convolutional layers.
                    for x in model_object.cnn_blocks.parameters():
                        x.requires_grad = False
                    print("The gradient flow in the convolutional layers has been disabled!")

                except Exception as e:
                    print("Error loading the trained model!")
                    print(e)
                    print("--- Exiting ---")
                    sys.exit()

            return model_object, loss_func, optim_func, lr_decay_func

        else:
            raise ValueError("The requested model does not exist! Try checking your spelling.")
