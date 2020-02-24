import numpy as np 
import tensorflow as tf 



class CapsNet:
	'''
	A Capsule neural network with dynamic routing with :-
		a) 3 Convolutional filter layers.
		b) 1 Primary Caps layer.
		c) 1 Digit Caps layer.
	'''

	def __init__(self, image_height, image_width, image_depth, learning_rate, decay_steps, decay_rate, primary_caps_vlength, digit_caps_vlength, epsilon, lambda_,
														m_plus, m_minus, reg_scale, routing_iteration, freeze_conv=False, input_placeholder=None):
		'''
		Initialize the CapsNet model with the fixed parameters.
		'''

		self.image_height			= image_height
		self.image_width			= image_width
		self.image_depth  			= image_depth
		self.learning_rate			= learning_rate
		self.decay_steps 		 	= decay_steps
		self.decay_rate				= decay_rate
		self.freeze_conv 			= freeze_conv
		self.primary_caps_vlength 	= primary_caps_vlength
		self.digit_caps_vlength 	= digit_caps_vlength
		self.epsilon 				= epsilon
		self.lambda_				= lambda_
		self.m_plus 				= m_plus 
		self.m_minus				= m_minus
		self.reg_scale				= reg_scale
		self.routing_iteration		= routing_iteration


		if input_placeholder is None:

			self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.image_height, self.image_width, self.image_depth), name='input')

		else:

			self.x = input_placeholder

		#keep prob has no use.
		self.keep_prob = tf.placeholder(tf.float32) #to be consistent with CNN model.

		self.build_model()
		self.reconstruction_network()
		self.total_loss()
		self.optimization()
		self.accuracy()


	def squash(self, capsule):
		'''
		CapsNet activation func.
		'''

		 #The output vector is in dimension -2 
		dot_product = tf.reduce_sum(tf.square(capsule), axis=-2, keepdims=True) 
		scalar_factor = dot_product/(1 + dot_product)/tf.sqrt(dot_product + self.epsilon)
		vec_squashed = scalar_factor * capsule
		return vec_squashed


	def routing(self, capsule_layer, num_capsules):
	
		#define a weight variable for one capsule first
		W = tf.get_variable('Weight', shape=(1, num_capsules, 2, self.primary_caps_vlength, self.digit_caps_vlength))
		b = tf.get_variable('Bias', shape=(1, 1, 2, self.digit_caps_vlength,1 )) 
		W = tf.tile(W, [tf.shape(capsule_layer)[0], 1, 1 ,1 ,1]) #tiling just makes a copy of the same weight variable for all the items in the batch. It is still the same weight.
		x = tf.tile(capsule_layer, [1, 1, 2, 1, 1]) #this is set up for dynamic routing later
		u_hat = tf.matmul(W,x, transpose_a=True) #[batch_size, 1152, 10, 16, 1]
		u_hat_stopped = tf.stop_gradient(u_hat, name='stopped_gradient')
		
		#coefficients for dynamic routing
		#MUST be initialized to zero for every round
		b_ij = tf.zeros([tf.shape(capsule_layer)[0], num_capsules, 2, 1, 1], dtype=tf.float32)
		
		
		for r_iter in range(self.routing_iteration):
			
			#softmax so that the value of the coefficient is between 0 and 1
			#lower level capsules that contribute the most to a particular higher level capsule
			#will have the largest value
			c_ij = tf.nn.softmax(b_ij, axis=2)
			
			#last iteration
			if r_iter == self.routing_iteration - 1:
				
				s_j = tf.multiply(c_ij, u_hat) #multiply the coefficients with each capsule
				
				#reducing the sum at axis 1 makes the capsules with highest coefficient to 
				#contribute more and the lowest coefficient capsules to contirbute less
				s_j = tf.reduce_sum(s_j, axis=1, keepdims=True) + b
				
				v_j = self.squash(s_j)
				
			else:
				
				s_j = tf.multiply(c_ij, u_hat_stopped)
				
				#reducing the sum at axis 1 makes the capsules with highest coefficient to contribute more and the 
				#lowest coefficient capsules to contirbute less
				s_j = tf.reduce_sum(s_j, axis=1, keepdims=True) + b 
				v_j = self.squash(s_j)
				
				#make a copy at the number of capsules axis in
				#order to find the scalar product
				v_j_tiled = tf.tile(v_j, [1, num_capsules, 1, 1, 1])
				
				product = u_hat_stopped * v_j_tiled 
				#by reducing the sum at axis 3, where the previous product produced new vectors, gives a scalar value.
				#Whichever capsules that agrees with each other will produce high valued vectors. Sum reduce would
				#add them all up together to bring a scalar value which then used for the softmax to enable the routing
				u_produce_v = tf.reduce_sum(product, axis=3, keepdims=True)
				
				#append the contribution of each lower level capsules to the higher level capsules
				b_ij += u_produce_v
		
		return v_j


	def build_model(self):
		'''
		CapsNet Architecture.
		'''

		self.y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='target')

		conv1 = tf.contrib.layers.conv2d(self.x, num_outputs=128, kernel_size=3, stride=2, padding='SAME', 
								 activation_fn=tf.nn.relu, scope='Conv_1', trainable=not self.freeze_conv)

		conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=256, kernel_size=5, stride=2, padding='SAME',
								 activation_fn=tf.nn.relu, scope='Conv_2', trainable=not self.freeze_conv)

		conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=256, kernel_size=5, stride=1, padding='SAME',
								 activation_fn=tf.nn.relu, scope='Conv_3', trainable=not self.freeze_conv)

		capsules = tf.reshape(conv3, (tf.shape(conv3)[0], -1, self.primary_caps_vlength, 1), name='FirstCaps')
		num_capsules = int((self.image_height//2**2)*(self.image_width//2**2)*256/self.primary_caps_vlength) #= 100352
		primary_caps = self.squash(capsules) #non-linearity

		primary_caps = tf.reshape(primary_caps, shape=(tf.shape(capsules)[0], -1, 1, self.primary_caps_vlength, 1 ),
						  name='PrimaryCapsules')

		digits = self.routing(primary_caps, num_capsules)

		self.digits = tf.squeeze(digits, axis=1, name='digits') # [batch_size, 10, 16, 1]

		#the length of each vectors in the digit capsule layer
		#[batch_size, 10, 1, 1]
		self.v_lengths = tf.sqrt(tf.reduce_sum(tf.square(self.digits), axis=2, keepdims=True) + self.epsilon, name='digit_vectors') 

		#Objective function for classification
		max_l = tf.square(tf.maximum(0., self.m_plus - self.v_lengths), name='max_l')
		max_r = tf.square(tf.maximum(0., self.v_lengths - self.m_minus), name='max_r')

		max_l = tf.reshape(max_l, shape=(tf.shape(self.digits)[0], -1))
		max_r = tf.reshape(max_r, shape=(tf.shape(self.digits)[0], -1))
		T_c = self.y

		L_c = T_c * max_l + self.lambda_*(1-T_c)*max_r
		self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name='margin_loss')

	def reconstruction_network(self):
		'''
		Reconstruction network.
		'''

		#First step is to mask out all the vectors from digit capsules except the correct class vector.
		#By performing an element-wise multiplication between the 2 capsules and the Y vector
		mask = tf.multiply(tf.squeeze(self.digits), tf.reshape(self.y, (-1, 2, 1)))
		vector_j = tf.reshape(mask, shape=(tf.shape(self.digits)[0], 2*self.digit_caps_vlength)) 
		fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512, activation_fn=tf.nn.relu, trainable=True)
		fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024, activation_fn=tf.nn.relu, trainable=True)
		self.decoded = tf.layers.dense(fc2, units=self.image_height*self.image_width, activation=tf.nn.sigmoid, name='decoded')

		#RECONSTUCTION LOSS
		origin   = tf.reshape(self.x, shape=(tf.shape(self.digits)[0], -1))
		squared  = tf.square(self.decoded - origin)

		#reg scale is used to control the total loss so that the margin loss would not be overtaken by the reconstruction
		#loss.
		self.reconst_err = self.reg_scale * tf.reduce_sum(squared, name='reconstruction_error')


	def total_loss(self):
		'''
		Total loss of the entire network (Classification + Reconstruction loss)
		'''

		self.loss = self.margin_loss + self.reconst_err #total for both classification and reconstruction


	def optimization(self):
		'''
		Model optimization function.
		'''

		#decayed learning rate
		global_step = tf.Variable(0, trainable=False)
		self.decayed_lr = tf.train.exponential_decay(self.learning_rate,
		                                            global_step, self.decay_steps,
		                                            0.98, staircase=True)


		self.train_step = tf.train.AdamOptimizer(self.decayed_lr).minimize(self.loss, global_step=global_step) 

	def accuracy(self):
		'''	
		Calculate model's accuracy.
		'''

		self.softmaxed_prediction = tf.nn.softmax(self.v_lengths, axis=1)
		argmax_idx = tf.argmax(self.softmaxed_prediction, axis=1)
		argmax_idx = tf.reshape(argmax_idx, shape=(tf.shape(self.x)[0],))
		correct_prediction = tf.equal(argmax_idx, tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'), name='accuracy')















