import argparse

defaults = {
	'epoch' : 30,
	'learning_rate' : 1e-4,
	'dropout_rate' : 0.5,
	'image_height': 64,
	'image_width' : 64,
	'image_depth' : 1,
	'batch_size' : 30,
	'test_data_path': './dataset/testing/2000/',
	'train_data_path' : './dataset/training/500/1_polygon/'
}



parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Specify the model. CNN or CapsNet', required=True)
parser.add_argument('--data_path', type=str, help='Specify the path to the data.', required=True)
parser.add_argument('--freeze_conv', type=bool, help='Convolutional layers should be freezed during the transfer learning.', required=True)
parser.add_argument('--epoch', type=int, default=defaults['epoch'], help='Specify the number of training epoch.')
parser.add_argument('--image_height', type=int, default=defaults['image_height'], help='Specify the height of the training images.')
parser.add_argument('--image_width', type=int, default=defaults['image_width'], help='Specify the width of the training images.')
parser.add_argument('--image_depth', type=int, default=defaults['image_depth'], help='Specify the depth of the training images.')
parser.add_argument('--learning_rate', type=float, default=defaults['learning_rate'], help='Specify the learning rate of the model.')
parser.add_argument('--dropout_rate', type=float, default=defaults['dropout_rate'], help='Specify the dropout rate for the training.')
parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='Specify the batch size for the training.')
parser.add_argument('--train_data_path', type=str, default=defaults['train_data_path'], help='Specify the folder for the training data.')
parser.add_argument('--test_data_path', type=str, default=defaults['test_data_path'], help='Specify the folder for the testing data.')

args = parser.parse_args()

assert args.model == 'CapsNet' or args.model == 'CNN', "Model argument must be either CapsNet or CNN."