import argparse
import sys


defaults = {
	'img_height'    		: 64,
	'img_width'     		: 64,
	'padding'       		: 10,
	'tri_ratio' 			: 0.25,
	'tri_rect_gap'			: 0.05,
	'min_arrow_len' 		: 32,
	'rect_thickness'		: 0.4,
	'num_of_data'       	        : 1000,
	'min_tri_len'			: 20,
	'max_tri_len'			: 50,
	'min_rect_length'		: 20,
	'max_rect_length'		: 62,
	'min_angle'			: 30,
	'max_angle'			: 160,
	'train_save_path'		: '../dataset/training/',
	'test_save_path'		: '../dataset/testing/',
	'mode'					: 'train'
}

parser = argparse.ArgumentParser()

parser.add_argument('--min_arrow_len', default=defaults['min_arrow_len'], type=int, help='Define the minimum length of the arrow.')
parser.add_argument('--height', default=defaults['img_height'], type=int, help='Define the image height')
parser.add_argument('--width', default=defaults['img_width'], type=int, help='Define the image width.')
parser.add_argument('--padding', default=defaults['padding'], type=int, help='Define the padding size. Padding is used to make sure the points generated are not too close to the image boundary.')
parser.add_argument('--tri_ratio', default=defaults['tri_ratio'], type=int, help='Define the ratio of the triangle"s height relative to the arrow"s length')
parser.add_argument('--tri_rect_gap', default=defaults['tri_rect_gap'], type=int, help='Define the ratio of the gap between the square and the triangle with relative to the length of the arrow.')
parser.add_argument('--rect_thickness', default=defaults['rect_thickness'], type=int, help='Define the thickness ratio of the rectangle with relative to the triangle"s base width.')
parser.add_argument('--num_of_data', default=defaults['num_of_data'], type=int, help='Define the number of data to be generated.')
parser.add_argument('--train_save_path', default=defaults['train_save_path'], type=str, help='Define the training save path.')
parser.add_argument('--test_save_path', default=defaults['test_save_path'], type=str, help='Define the testing save path.')
parser.add_argument('--min_tri_len', default=defaults['min_tri_len'], type=int, help='Define the minimum length for the triangle.')
parser.add_argument('--max_tri_len', default=defaults['max_tri_len'], type=int, help='Define the maximum length for the triangle.')
parser.add_argument('--max_rect_length', default=defaults['max_rect_length'], type=int, help='Define the maximum length for the rectangle.')
parser.add_argument('--min_rect_length', default=defaults['min_rect_length'], type=int, help='Define the minimum length for the rectangle.')
parser.add_argument('--min_angle', default=defaults['min_angle'], type=int, help='Define the minimum angle between triangle vector and rectangle vector.')
parser.add_argument('--max_angle', default=defaults['max_angle'], type=int, help='Define the maximum angle between triangle vector and rectangle vector.')
parser.add_argument('--mode', default=defaults['mode'], type=str, help='Define whether the data generated are for testing purposes or training purposes. train or test.')

cfg = parser.parse_args()


# assert cfg.num_of_sides >= 3, "Number of sides must be more than 2 !"
# assert cfg.height >= 20, "Image height must be at least 20!"
# assert cfg.width >= 20, "Image width must be at least 20!"
# assert cfg.padding*2 <= cfg.width or cfg.padding*2 <= cfg.height, "Padding size cannot exceed half of the width nor height of the image!"
# assert cfg.min_side_length < cfg.height or cfg.min_side_length < cfg.width, "The length of the sides cannot exceed the width nor the height of the image!"
# assert cfg.min_angle > 0, "The minimum angle must be more than 0!"
# assert cfg.max_angle < 180, "The maximum angle must be less than 180!"
# assert cfg.max_iter > 0, "The maximum iteration value must be more than 0!"
