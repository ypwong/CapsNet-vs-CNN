import cv2
import os
import arrows
import non_arrows

from cfg import cfg


#Configurations
image_height 			= cfg.height
image_width  			= cfg.width
padding 	 			= cfg.padding 
tri_height_ratio		= cfg.tri_ratio
tri_rect_gap 		 	= cfg.tri_rect_gap
min_arrow_len 			= cfg.min_arrow_len 
rect_thickness 			= cfg.rect_thickness
num_of_data 			= cfg.num_of_data
train_save_path 		= cfg.train_save_path
test_save_path 			= cfg.test_save_path
min_tri_len 			= cfg.min_tri_len
max_tri_len 			= cfg.max_tri_len
min_rect_length 		= cfg.min_rect_length
max_rect_length 		= cfg.max_rect_length
min_angle 				= cfg.min_angle 
max_angle 				= cfg.max_angle
mode 					= cfg.mode

#init classes
arrow_gen = arrows.Arrows(image_height			= image_height,
						  image_width 			= image_width,
						  padding    		 	= padding,
						  tri_height_ratio 		= tri_height_ratio,
						  tri_rect_gap			= tri_rect_gap,
						  min_arrow_len			= min_arrow_len,
						  rect_thickness		= rect_thickness)

non_arrow_gen = non_arrows.NonArrows(image_height		= image_height,
									 image_width		= image_width,
									 min_tri_len		= min_tri_len,
									 max_tri_len		= max_tri_len,
									 max_rect_length	= max_rect_length,
									 min_rect_length 	= min_rect_length,
									 min_angle			= min_angle,
									 max_angle			= max_angle,
									 padding			= padding,
									 tri_ratio 			= tri_height_ratio,
									 rect_thickness		= rect_thickness)




if mode == 'train':

	save_path = train_save_path


	non_arrow_save_path = save_path+str(num_of_data)+'/2_polygons/non_arrow/'
	triangle_save_path	 = save_path+str(num_of_data)+'/1_polygon/triangle/'
	rectangle_save_path	 = save_path+str(num_of_data)+'/1_polygon/rectangle/'
	arrow_save_path     = save_path+str(num_of_data)+'/2_polygons/arrow/'

	if not os.path.exists(non_arrow_save_path) : os.makedirs(non_arrow_save_path)
	if not os.path.exists(triangle_save_path) : os.makedirs(triangle_save_path)
	if not os.path.exists(rectangle_save_path) : os.makedirs(rectangle_save_path)
	if not os.path.exists(arrow_save_path) : os.makedirs(arrow_save_path)

	for i in range(num_of_data//2):


		arrow		= arrow_gen.generate_polygon()
		triangle	= arrow_gen.generate_polygon(rectangle=False)
		rectangle	= arrow_gen.generate_polygon(triangle=False)
		non_arrow 	= non_arrow_gen.generate_polygon()

		cv2.imwrite(arrow_save_path+str(i)+'.jpg', arrow)
		cv2.imwrite(triangle_save_path+str(i)+'.jpg', triangle)
		cv2.imwrite(rectangle_save_path+str(i)+'.jpg', rectangle)
		cv2.imwrite(non_arrow_save_path+str(i)+'.jpg', non_arrow)

if mode == 'test':

	save_path = test_save_path
	num_of_data = 2000 #we use 1000 images in each class for evaluation

	non_arrow_save_path = save_path+str(num_of_data)+'/2_polygons/non_arrow/'
	triangle_save_path	 = save_path+str(num_of_data)+'/1_polygon/triangle/'
	rectangle_save_path	 = save_path+str(num_of_data)+'/1_polygon/rectangle/'
	arrow_save_path     = save_path+str(num_of_data)+'/2_polygons/arrow/'

	if not os.path.exists(non_arrow_save_path) : os.makedirs(non_arrow_save_path)
	if not os.path.exists(triangle_save_path) : os.makedirs(triangle_save_path)
	if not os.path.exists(rectangle_save_path) : os.makedirs(rectangle_save_path)
	if not os.path.exists(arrow_save_path) : os.makedirs(arrow_save_path)

	for i in range(num_of_data//2):


		arrow		= arrow_gen.generate_polygon()
		triangle	= arrow_gen.generate_polygon(rectangle=False)
		rectangle	= arrow_gen.generate_polygon(triangle=False)
		non_arrow 	= non_arrow_gen.generate_polygon()

		cv2.imwrite(arrow_save_path+str(i)+'.jpg', arrow)
		cv2.imwrite(triangle_save_path+str(i)+'.jpg', triangle)
		cv2.imwrite(rectangle_save_path+str(i)+'.jpg', rectangle)
		cv2.imwrite(non_arrow_save_path+str(i)+'.jpg', non_arrow)



