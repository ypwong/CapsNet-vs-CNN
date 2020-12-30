'''
Arrows-NonArrows Dataset Generator.
'''

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import cv2
import math
import random
from meta_classes import DataMeta #interface


class Arrow_vs_NonArrows(DataMeta):
	'''
	Arrow and Non-Arrow polygon logic.
	'''

	def __init__(self, **kwargs):
		'''
		Inititalize.
		'''

		print(kwargs)
		self.arrow_obj = Arrows(image_height=kwargs['image_height']//2,
								image_width=kwargs['image_width']//2,
								padding=kwargs['padding']//2,
								tri_height_ratio=kwargs['tri_height_ratio']//2,
								tri_rect_gap=kwargs['tri_rect_gap']//2,
								min_arrow_len=kwargs['min_arrow_len']//2,
								rect_thickness=kwargs['rect_thickness']//2)

		self.nonarrow_obj = NonArrows(image_height=kwargs['image_width']//2,
									  image_width=kwargs['image_width']//2,
									  min_tri_len=kwargs['min_tri_len']//2,
									  max_tri_len=kwargs['max_tri_len']//2,
									  min_angle=kwargs['min_angle']//2,
									  max_angle=kwargs['max_angle']//2,
									  padding=kwargs['padding']//2,
									  tri_height_ratio=kwargs['tri_height_ratio']//2,
									  min_rect_length=kwargs['min_rect_length']//2,
									  max_rect_length=kwargs['max_rect_length']//2,
									  rect_thickness=kwargs['rect_thickness']//2)


	def get_feature_level_data(self):
		'''
		Return the names and indices of the feature-level data.
		'''
		return {
			'triangle' : 0,
			'rectangle': 1
		}

	def get_object_level_data(self):
		'''
		Return the names and indices of the object-level data.
		'''
		return {
			'arrow': 0,
			'nonarrow': 1
		}

	def gen_feature_level(self, feature_index):
		'''
		Returns a single feature-level image based on the given index of the feature.
		'''
		if feature_index == 0:
			return self.arrow_obj.generate_polygon(rectangle=False)
		elif feature_index == 1:
			return self.arrow_obj.generate_polygon(triangle=False)
		else:
			raise ValueError("The given index for the class of feature-level data does not exist! HINT: check the available \
							classes with 'get_feature_level_data' method")

	def gen_object_level(self, object_index):
		'''
		Returns a single object-level image based on the given index of the object.
		'''
		if object_index == 0:
			return self.arrow_obj.generate_polygon()
		elif object_index == 1:
			return self.nonarrow_obj.generate_polygon()
		else:
			raise ValueError("The given index for the class of object-level data does not exist! HINT: check the available \
							classes with 'get_object_level_data' method")



class Arrows:
	'''
	To generate arrows.
	'''

	def __init__(self, image_height, image_width, padding, tri_height_ratio,
						tri_rect_gap, min_arrow_len, rect_thickness):
		'''
		Initialize the fixed parameters for the arrow dataset.
		'''
		self.image_height 		= image_height
		self.image_width		= image_width
		self.padding 			= padding
		self.tri_height_ratio 	= tri_height_ratio
		self.tri_rect_gap 		= tri_rect_gap
		self.min_arrow_len 		= min_arrow_len
		self.rect_thickness 	= rect_thickness
		self.rect_ratio 		= self.tri_height_ratio + self.tri_rect_gap




	def list2numpy(self, given_list):
		'''
		Convert the given list into a np array
		'''

		return np.asarray(given_list, dtype='int32')


	def numpy2list(self, given_array):
		'''
		Convert the given array into a python list
		'''

		return list(given_array)


	def generate_points(self):
		'''
		Generate two points randomly based on the given criterias.
		'''

		length = 0

		while length < self.min_arrow_len: #keep generating points until the distance between the points is more than the minimum value.self.

			#generate two points within the image's size + padding.
			point1 = [random.randint(self.padding, self.image_width-self.padding), random.randint(self.padding, self.image_height-self.padding)]
			point2 = [random.randint(self.padding, self.image_width-self.padding), random.randint(self.padding, self.image_height-self.padding)]

			#vector from point 1 to point 2
			vect = self.list2numpy(point1) - self.list2numpy(point2)

			#length of the vector (i.e. distance between the two generated points)
			length = np.linalg.norm(vect)

		return point1, point2, length


	def find_point_on_line(self, point1, point2, ratio):
		'''
		Find the point of triangle's base on the given two points.
		'''

		original_line = self.list2numpy(point1) - self.list2numpy(point2)

		poly_vect = ratio*original_line

		poly_length = np.linalg.norm(poly_vect)

		new_point = self.list2numpy(point2) + poly_vect

		return new_point, poly_length


	def find_triangle_width(self, length):
		'''
		To find the width of the triangle's base.
		'''
		return math.tan(math.radians(30)) * length


	def find_perpendicular_vector(self, point1, point2):
		'''
		Find the vector that is perpendicular to the original vector in order to draw the triangle and the rectangle.
		'''

		ori_line = self.list2numpy(point1) - self.list2numpy(point2)

		perpendicular_vector = ori_line[[1,0]] #reverse the position
		perpendicular_vector[1] = -1*perpendicular_vector[1]

		return perpendicular_vector


	def triangle_points(self, perpendicular_vector, triangle_point, triangle_base_length):
		'''
		Get the points to draw the triangle.
		'''

		perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
		norm_perp_vector = perpendicular_vector/perpendicular_vector_length

		point1 = triangle_point + (norm_perp_vector)*triangle_base_length
		point2 = triangle_point + (-1*(norm_perp_vector))*triangle_base_length

		return point1, point2


	def rectangle_points(self, triangle_base_length, rectangle_point_on_line, point1_ori, perpendicular_vector,rect_thickness):
		'''
		Get the points to draw the rectangle.
		'''

		perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
		norm_perp_vector = perpendicular_vector/perpendicular_vector_length

		rectangle_base_length = triangle_base_length * rect_thickness

		point1 = point1_ori + (norm_perp_vector*rectangle_base_length)
		point2 = point1_ori + (-1*norm_perp_vector)*rectangle_base_length
		point3 = rectangle_point_on_line + (norm_perp_vector*rectangle_base_length)
		point4 = rectangle_point_on_line + (-1*norm_perp_vector)*rectangle_base_length


		return point1, point2, point3, point4


	def draw_triangle(self, point1, point2, point3, image):
		'''
		Draw the triangle on the given image.
		'''

		point1 = self.numpy2list(np.asarray(np.round(point1),dtype='int32'))
		point2 = self.numpy2list(np.asarray(np.round(point2),dtype='int32'))
		point3 = self.numpy2list(np.asarray(np.round(point3),dtype='int32'))


		cv2.line(image, (point1[0], point1[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		cv2.line(image, (point1[0], point1[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		cv2.line(image, (point3[0], point3[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		return None


	def draw_rectangle(self,point1, point2, point3, point4, image):
		'''
		Draw the rectangle on the given image.
		'''

		point1 = self.numpy2list(np.asarray(np.round(point1),dtype='int32'))
		point2 = self.numpy2list(np.asarray(np.round(point2),dtype='int32'))
		point3 = self.numpy2list(np.asarray(np.round(point3),dtype='int32'))
		point4 = self.numpy2list(np.asarray(np.round(point4),dtype='int32'))

		cv2.line(image, (point1[0], point1[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		cv2.line(image, (point1[0], point1[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		cv2.line(image, (point2[0], point2[1]),
						(point4[0], point4[1]),
						(255,255,255), 1)

		cv2.line(image, (point4[0], point4[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		return None


	def generate_polygon(self, triangle=True, rectangle=True):
		'''
		Generate an arrow.
		'''

		image = np.zeros((self.image_height, self.image_width, 1)) #Initialize the image.

		point1, point2, line_length = self.generate_points() #Generate two random points and get the length of the vector.

		#Get the point on the line where the triangle's base would intersect and the height of the triangle.


		if triangle and not rectangle:

			adjusted_size_tri = self.tri_height_ratio + 0.15
		else:
			adjusted_size_tri = self.tri_height_ratio

		triangle_point, tri_height = self.find_point_on_line(point1, point2, adjusted_size_tri)

		triangle_base_length = self.find_triangle_width(tri_height) #Get the length of the triangle's base.

		perpendicular_vector = self.find_perpendicular_vector(point1, point2) #Find the vector that is perpendicular to the original generated vector.

		#Get the rest of the points needed to draw triangle.
		point3, point4 = self.triangle_points(perpendicular_vector, triangle_point, triangle_base_length)

		if rectangle and not triangle:

			adjusted_size_rect = self.rect_ratio - 0.15

		else:
			adjusted_size_rect = self.rect_ratio

		#Get the point on the original vector to draw the rectangle.
		rectangle_point_on_line, _ = self.find_point_on_line(point1, point2, adjusted_size_rect)

		#Get the rest of the points needed to draw the rectangle.
		point5, point6, point7, point8 = self.rectangle_points(triangle_base_length, rectangle_point_on_line, point1, perpendicular_vector, self.rect_thickness)

		if triangle:
			self.draw_triangle(point2, point3, point4, image) #Draw the triangle on the image.

		if rectangle:
			self.draw_rectangle(point5, point6, point7, point8, image) #Draw the rectangle on the image.

		return image



class NonArrows:
	'''
	To generate non-arrows.
	'''

	def __init__(self, image_height, image_width, min_tri_len, max_tri_len, min_angle,
											max_angle, padding, tri_height_ratio, min_rect_length,
																max_rect_length,rect_thickness):
		'''
		Initialize the fixed parameters.
		'''

		self.image_height 		= image_height
		self.image_width  		= image_width
		self.min_tri_length  	= min_tri_len
		self.max_tri_length  	= max_tri_len
		self.max_rect_length	= max_rect_length,
		self.min_rect_length	= min_rect_length
		self.min_angle	  		= min_angle
		self.max_angle	  		= max_angle
		self.padding 			= padding
		self.tri_height_ratio 	= tri_height_ratio
		self.rect_thickness 	= rect_thickness



	def list2numpy(self, given_list):
		'''
		Convert the given list into a np array
		'''

		return np.asarray(given_list, dtype='int32')

	def numpy2list(self, given_array):
		'''
		Convert the given array into a python list
		'''

		return list(given_array)


	def check_angle(self, point1, point2, point3):
		'''
		Check the angle between the two lines.
		'''

		first_vector  = self.list2numpy(point1) - self.list2numpy(point2)
		second_vector = self.list2numpy(point3) - self.list2numpy(point2)

		rad_angle = math.acos((first_vector.dot(second_vector))/(np.linalg.norm(first_vector)*np.linalg.norm(second_vector) + 1e-8))

		return math.degrees(rad_angle)

	def find_perpendicular_vector(self, point2, point3):
		'''
		Find the point of triangle's base on the given two points.
		'''

		ori_line = self.list2numpy(point2) - self.list2numpy(point3)

		perpendicular_vector = ori_line[[1,0]]
		perpendicular_vector[1] = -1*perpendicular_vector[1]

		return perpendicular_vector


	def find_triangle_width(self, length):
		'''
		Calculate the width of the triangle based on the height.
		'''
		return math.tan(math.radians(30)) * length


	def rectangle_points(self, triangle_base_length, rectangle_point_on_line, point1_ori, perpendicular_vector,rect_thickness):
		'''
		Get the points of the rectangle.
		'''

		perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
		norm_perp_vector = perpendicular_vector/perpendicular_vector_length

		rectangle_base_length = triangle_base_length * rect_thickness

		point1 = point1_ori + (norm_perp_vector*rectangle_base_length)
		point2 = point1_ori + (-1*norm_perp_vector)*rectangle_base_length
		point3 = rectangle_point_on_line + (norm_perp_vector*rectangle_base_length)
		point4 = rectangle_point_on_line + (-1*norm_perp_vector)*rectangle_base_length


		return point1, point2, point3, point4


	def triangle_points(self, perpendicular_vector, triangle_point, triangle_base_length):
		'''
		Get the points of the triangle.
		'''

		perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
		norm_perp_vector = perpendicular_vector/perpendicular_vector_length

		point1 = triangle_point + (norm_perp_vector)*triangle_base_length
		point2 = triangle_point + (-1*(norm_perp_vector))*triangle_base_length

		return point1, point2

	def find_point_on_line(self, point1, point2, ratio):
		'''
		Get the
		'''

		ori_line = self.list2numpy(point1) - self.list2numpy(point2)

		poly_vect = ratio*ori_line

		poly_length = np.linalg.norm(poly_vect)

		new_point = self.list2numpy(point2) + poly_vect

		return new_point, poly_length


	def generate_points(self):
		'''
		Generate three points based on the given criteria(s).
		'''
		length1 = 0
		length2 = 0
		angle = 0
		while ((length1 < self.min_rect_length or length1 > self.max_rect_length) or (length2 < self.min_tri_length or length2 > self.max_tri_length)
				or (angle<self.min_angle or angle > self.max_angle)):

			point1 = [random.randint(self.padding, self.image_width-self.padding), random.randint(self.padding, self.image_height-self.padding)]
			point2 = [random.randint(self.padding, self.image_width-self.padding), random.randint(self.padding, self.image_height-self.padding)]
			point3 = [random.randint(self.padding, self.image_width-self.padding), random.randint(self.padding, self.image_height-self.padding)]


			vect1 = self.list2numpy(point1) - self.list2numpy(point2)
			vect2 = self.list2numpy(point3) - self.list2numpy(point2)

			length1 = np.linalg.norm(vect1)
			length2 = np.linalg.norm(vect2)

			angle = self.check_angle(point1, point2, point3)

		return point1, point2, point3, angle

	def draw_triangle(self, point1, point2, point3, image):
		'''
		Draw triangle on the given image.
		'''

		point1 = self.numpy2list(np.asarray(np.round(point1),dtype='int32'))
		point2 = self.numpy2list(np.asarray(np.round(point2),dtype='int32'))
		point3 = self.numpy2list(np.asarray(np.round(point3),dtype='int32'))


		cv2.line(image, (point1[0], point1[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		cv2.line(image, (point1[0], point1[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		cv2.line(image, (point3[0], point3[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		return None

	def draw_rectangle(self, point1, point2, point3, point4, image):
		'''
		Draw rectangle on the given image.
		'''

		point1 = self.numpy2list(np.asarray(np.round(point1),dtype='int32'))
		point2 = self.numpy2list(np.asarray(np.round(point2),dtype='int32'))
		point3 = self.numpy2list(np.asarray(np.round(point3),dtype='int32'))
		point4 = self.numpy2list(np.asarray(np.round(point4),dtype='int32'))

		cv2.line(image, (point1[0], point1[1]),
						(point2[0], point2[1]),
						(255,255,255), 1)

		cv2.line(image, (point1[0], point1[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		cv2.line(image, (point2[0], point2[1]),
						(point4[0], point4[1]),
						(255,255,255), 1)

		cv2.line(image, (point4[0], point4[1]),
						(point3[0], point3[1]),
						(255,255,255), 1)

		return None


	def generate_polygon(self):
		'''
		Generate Non-Arrow polygons.
		'''
		image = np.zeros((self.image_height, self.image_width,1))

		point1, point2, point3, angle = self.generate_points()

		triangle_point, tri_height 	= self.find_point_on_line(point2, point3, self.tri_height_ratio)

		triangle_base_length = self.find_triangle_width(tri_height)

		perpendicular_vector = self.find_perpendicular_vector(point2, point3)

		point4, point5 = self.triangle_points(perpendicular_vector, triangle_point, triangle_base_length)

		perpendicular_vector = self.find_perpendicular_vector(point1, point2)

		point6, point7, point8, point9 = self.rectangle_points(triangle_base_length, point2, point1, perpendicular_vector, self.rect_thickness)

		self.draw_triangle(point3, point4, point5, image)

		self.draw_rectangle(point6, point7, point8, point9, image)

		return image
