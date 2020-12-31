'''
General configurations.
'''

import argparse
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SIMPLE_CNN_FEATURE_MODEL_PATH = './trained_models/simple_cnn_feature.pth'
SIMPLE_CNN_OBJECT_MODEL_PATH = './trained_models/simple_cnn_object.pth'


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Specify the model that you want to train.', required=True)
parser.add_argument('--mode', type=str, default=None, help='Specify the mode of the training. None or transfer_learning.')
parser.add_argument('--model_load_path', type=str, default=None,
                    help='Specify the path of the model to be loaded if the mode of training is transfer_learning')
parser.add_argument('--dataset_name', type=str, help='Specify the name of the dataset for this experiment.', required=True)
parser.add_argument('--dataset_level', type=str, help='Either feature_level or object_level', required=True)
parser.add_argument('--train_num_data_per_class', type=int, default=1000, help='Specify the number of data to be generated per class for training.')
parser.add_argument('--test_num_data_per_class', type=int, default=1000, help='Specify the number of data to be generated per class for testing.')
parser.add_argument('--load_from_disk', type=bool, help='Setting to True would load the dataset from disk. Else dataset will be \
                        generated and written to disk.')
parser.add_argument('--data_path', type=str, default='./generated_dataset/', help='Specify the path to the dataset folder to write to or load from.',
                    required=True)
parser.add_argument('--batch_size', type=int, default=10, help='Specify the batch size of the data to be fed to the models.')
parser.add_argument('--num_workers', type=int, default=3, help='Specify the num of workers to be used to generate/load the data.')
parser.add_argument('--shuffle', type=bool, default=True, help='Setting this to True will enable shuffling among the dataset.')
parser.add_argument('--epoch', type=int, default=100, help='Specify the number of epochs for the operation.')

args = parser.parse_args()

assert args.mode is None or args.mode == 'transfer_learning', "Error! The specified mode does not exist. It is either None or transfer_learning."
