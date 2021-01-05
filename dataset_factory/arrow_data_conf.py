'''
Configuration for the arrow vs non-arrow dataset.
'''

defaults = {
	'image_height'    		: 64,
	'image_width'     		: 64,
	'padding'       		: 10,
	'tri_height_ratio' 		: 0.25,
	'tri_rect_gap'			: 0.05,
	'min_arrow_len' 		: 32,
	'rect_thickness'		: 0.4,
	'min_tri_len'			: 20,
	'max_tri_len'			: 50,
	'min_rect_length'		: 20,
	'max_rect_length'		: 62,
	'min_angle'				: 30,
	'max_angle'				: 160,
	'train_save_path'		: '../dataset/training/',
	'test_save_path'		: '../dataset/testing/'
}
