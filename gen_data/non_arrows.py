import numpy as np 
import cv2
import math
import random



class NonArrows:
	'''
	To generate non-arrows.
	'''

	def __init__(self, image_height, image_width, min_tri_len, max_tri_len, min_angle, 
											max_angle, padding, tri_ratio, min_rect_length, 
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
		self.tri_ratio 			= tri_ratio
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

		triangle_point, tri_height 	= self.find_point_on_line(point2, point3, self.tri_ratio)

		triangle_base_length = self.find_triangle_width(tri_height)

		perpendicular_vector = self.find_perpendicular_vector(point2, point3)

		point4, point5 = self.triangle_points(perpendicular_vector, triangle_point, triangle_base_length)

		perpendicular_vector = self.find_perpendicular_vector(point1, point2)

		point6, point7, point8, point9 = self.rectangle_points(triangle_base_length, point2, point1, perpendicular_vector, self.rect_thickness)

		self.draw_triangle(point3, point4, point5, image)

		self.draw_rectangle(point6, point7, point8, point9, image)

		return image



