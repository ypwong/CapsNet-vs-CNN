import numpy as np 
import tensorflow as tf 


class CNN:
	'''
	A convolutional neural network with :-
		a) 3 Convolutional filter layers.
		b) Max Pooling between each convolutional layers.
		c) 2 Fully-Connected layers
	'''

	def __init__(self, image_height, image_width, image_depth, learning_rate, decay_steps, decay_rate, freeze_conv=False, input_placeholder=None):
		'''
		Initialize the model parameters.
		'''

		self.image_height 	= image_height
		self.image_width  	= image_width
		self.image_depth  	= image_depth
		self.learning_rate	= learning_rate
		self.freeze_conv 	= freeze_conv
		self.decay_steps 	= decay_steps
		self.decay_rate		= decay_rate

		#Declare the input placeholder
		if input_placeholder is None : 

			self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.image_height, self.image_width, self.image_depth), name='inputs')
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

		else:
			self.x = input_placeholder
			self.keep_prob = tf.constant(1, dtype=tf.float32, name='keep_prob')

		self.build_model()
		self.loss_func()
		self.optimization()
		self.accuracy()


	def max_pool_2x2(self, x):
		'''
		Max-Pooling layer
		'''
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')


	def weight_variable(self, shape):
		'''
		Define a weight variable in the given shape.
		'''
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		'''
		Define a bias variable in the given shape.
		'''
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def build_model(self):
		'''
		Build of the model.
		'''

		#label
		self.y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='target')

		conv1 = tf.contrib.layers.conv2d(self.x, num_outputs=128, kernel_size=3, stride=1, padding='SAME', 
											activation_fn=tf.nn.relu, scope='Conv_1', trainable=not self.freeze_conv)

		conv1_pool = self.max_pool_2x2(conv1)

		conv2 = tf.contrib.layers.conv2d(conv1_pool, num_outputs=256, kernel_size=5, stride=1, padding='SAME', 
											activation_fn=tf.nn.relu, scope='Conv_2', trainable=not self.freeze_conv)

		conv2_pool = self.max_pool_2x2(conv2)

		conv3 = tf.contrib.layers.conv2d(conv2_pool, num_outputs=256, kernel_size=5, stride=1, padding='SAME', 
											activation_fn=tf.nn.relu, scope='Conv_3', trainable=not self.freeze_conv)

		conv3_pool = self.max_pool_2x2(conv3)

		reshape_size = (self.image_height//2**3) * (self.image_width//2**3) * 256 #feature vector's size

		feature_vector = tf.reshape(conv3_pool, [-1, reshape_size])

		fully_connected1 = tf.nn.relu(tf.matmul(feature_vector, self.weight_variable([reshape_size, 128])) + self.bias_variable([128]))

		dropout_layer = tf.nn.dropout(fully_connected1, self.keep_prob)

		#logits
		self.logits = tf.matmul(dropout_layer, self.weight_variable([128, 2])) + self.bias_variable([2])

		self.softmaxed_prediction = tf.nn.softmax(self.logits, axis=1)


	def loss_func(self):
		'''
		Loss function for the model.
		'''

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits), name='loss')

	def optimization(self):
		'''
		Model optimization functions
		'''

		global_step = tf.Variable(0, trainable=False)
		self.decayed_lr = tf.train.exponential_decay(self.learning_rate,
                                            global_step, self.decay_steps,
                                            self.decay_rate, staircase=True)

		self.train_step = tf.train.AdamOptimizer(self.decayed_lr).minimize(self.loss, global_step=global_step)

	def accuracy(self):
		'''
		Calculate model's accuracy.
		'''

		correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')












