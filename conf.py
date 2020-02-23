import argparse

defaults = {
	'epoch' : 30,
	'learning_rate' : 1e-4,
	'dropout_rate' : 0.5,
	'image_height': 64,
	'image_width' : 64,
	'image_depth' : 1,
	'batch_size' : 30,
	'primary_caps_vlength' : 16,
	'digit_caps_vlength' : 32,
	'epsilon' : 1e-8,
	'lambda_': 0.5,
	'm_plus' : 0.9,
	'm_minus' : 0.1,
	'reg_scale': 0.005,
	'routing_iteration' : 3,
	'decay_steps' : 50,
	'decay_rate' : 0.98,
	'test_data_path_base': './dataset/testing/2000/',
	'train_data_path' : './dataset/training/500/1_polygon/',
	'fig_save_path' : './figures/'
}



parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Specify the model. CNN or CapsNet', required=True)
parser.add_argument('--train_data_path', type=str, default=defaults['train_data_path'], help='Specify the folder for the training data.')
parser.add_argument('--freeze_conv', type=bool, help='Convolutional layers should be freezed during the transfer learning.', required=True)
parser.add_argument('--epoch', type=int, default=defaults['epoch'], help='Specify the number of training epoch.')
parser.add_argument('--image_height', type=int, default=defaults['image_height'], help='Specify the height of the training images.')
parser.add_argument('--image_width', type=int, default=defaults['image_width'], help='Specify the width of the training images.')
parser.add_argument('--image_depth', type=int, default=defaults['image_depth'], help='Specify the depth of the training images.')
parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='Specify the batch size for model training.')
parser.add_argument('--learning_rate', type=float, default=defaults['learning_rate'], help='Specify the learning rate of the model.')
parser.add_argument('--dropout_rate', type=float, default=defaults['dropout_rate'], help='Specify the dropout rate for the training.')
parser.add_argument('--primary_caps_vlength', type=int, default=defaults['primary_caps_vlength'], help='Specify the length of the primary capsule vectors.')
parser.add_argument('--digit_caps_vlength', type=int, default=defaults['digit_caps_vlength'], help='Specify the length of the digits capsule vectors.')
parser.add_argument('--epsilon', type=float, default=defaults['epsilon'], help='Specify the epsilon value for CapsNet.')
parser.add_argument('--lambda_', type=float, default=defaults['lambda_'], help='Specify the lambda value for CapsNet.')
parser.add_argument('--m_plus', type=float, default=defaults['m_plus'], help='Specify the m plus value for CapsNet.')
parser.add_argument('--m_minus', type=float, default=defaults['m_minus'],help='Specify the m minus value for CapsNet.')
parser.add_argument('--reg_scale', type=float, default=defaults['reg_scale'], help='Specify the regularization scale value for CapsNet"s recon network.')
parser.add_argument('--routing_iteration', type=float, default=defaults['routing_iteration'], help='Specify the routing iteration for CapsNet.')
parser.add_argument('--decay_steps', type=int, default=defaults['decay_steps'], help='Specify the decay steps for the models.')
parser.add_argument('--decay_rate', type=float, default=defaults['decay_rate'], help='Specify the rate of decay for the models.')
parser.add_argument('--test_data_path_base', type=str, default=defaults['test_data_path_base'], help='Specify the folder for the testing data.')
parser.add_argument('--fig_save_path', type=str, default=defaults['fig_save_path'], help='Specify the folder for saving the result figures.')

args = parser.parse_args()

assert args.model == 'CapsNet' or args.model == 'CNN', "Model argument must be either CapsNet or CNN."