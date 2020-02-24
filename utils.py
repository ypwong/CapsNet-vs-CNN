import os
import glob
from random import shuffle
import cv2
import numpy as np

class Utils:
	'''
	To prepare the dataset.
	'''

	def __init__(self, data_path, image_height, image_width, image_depth):
		'''
		Initialize the parameters.
		'''

		self.data_path = data_path
		self.image_height = image_height
		self.image_width = image_width
		self.image_depth = image_depth
		self.mode = self.data_path.split('/')[-2]

		self.classes_dict = {
			'1_polygon' :{'triangle':0, 'rectangle':1},
			'2_polygons':{'arrow':0, 'non_arrow':1}
		}
		self.inverse_classes_dict = {
			'1_polygon' :{0:'triangle', 1:'rectangle'},
			'2_polygons':{0:'arrow', 1:'non_arrow'}
		}

		data_list = self.get_data_list()

		self.total_data = len(data_list)

		self.images, self.labels = self.load_images(data_list)


	def get_data_list(self):
		'''
		Create a list containing the path of images and their corresponding labels.
		'''

		data_list = []


		#iterate through all the items in the given directory recursively.
		for path in glob.glob(self.data_path + '**', recursive=True):


			if path[-3:] == 'jpg':

				temp = [path, self.classes_dict[self.mode][path.split('/')[-2]]]
				data_list.append(temp)

		shuffle(data_list) #randomly shuffle the list

		return data_list

	def one_hot_encoder(self, label):
		'''
		Change the label integer into one hot vector label.
		'''
		one_hot = np.zeros((2))
		one_hot[label] = 1.0

		return one_hot



	def load_images(self, data_list):
		'''
		Read the images from the paths and save them as arrays.
		'''
		images_list = []
		labels_list = []

		for image_path, label in data_list:

			im_ = cv2.imread(image_path, 0)
			im_ = cv2.resize(im_, (self.image_width, self.image_height))
			im_ = np.asarray(im_, dtype='float32')/255
			im_ = np.reshape(im_, (self.image_width, self.image_height, 1))

			images_list.append(im_)

			one_hot_label = self.one_hot_encoder(label)

			labels_list.append(one_hot_label)

		images = np.asarray(images_list, dtype='float32')
		labels = np.asarray(labels_list, dtype='float32')

		return images, labels


	def __call__(self, first_index, last_index):
		'''
		Returns a batch of dataset based on the given indexes.
		'''

		#if the last index exceeds the total amount of data.
		if last_index > self.total_data : last_index = self.total_data

		batch_images, batch_labels = [], []

		for i in range(first_index, last_index):

			batch_images.append(self.images[i])
			batch_labels.append(self.labels[i])

		batch_images = np.asarray(batch_images, dtype='float32')
		batch_labels = np.asarray(batch_labels, dtype='float32')

		return batch_images, batch_labels










