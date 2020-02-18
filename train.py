import tensorflow as tf 
import numpy as  np 
import utils
from conf import args as cfg
import CNN
import CapsNet
from tqdm import tqdm
from tabulate import tabulate

tf.reset_default_graph()
sess = tf.InteractiveSession()

if cfg.model == 'CNN':

	model = CNN.CNN(image_height=cfg.image_height,
					image_width=cfg.image_width,
					image_depth= cfg.image_depth,
					learning_rate=cfg.learning_rate
					)


# if cfg.model == 'CapsNet':

# 	model = CapsNet.CapsNet()


utils_train = utils.Utils(data_path=cfg.train_data_path,
					image_height=cfg.image_height,
					image_width=cfg.image_width,
					image_depth=cfg.image_depth)

utils_test = utils.Utils(data_path= cfg.test_data_path,
						image_height=cfg.image_height,
						image_width=cfg.image_width,
						image_depth=cfg.image_depth)

total_training_data = utils_train.total_data
total_testing_data  = utils_test.total_data

save_model_path = './model_ckpt/'+cfg.model+'/'+str(cfg.data_path.split('/')[-1])
load_model_path = None

if cfg.freeze_conv:

	load_model_path = './model_ckpt/CNN/'+str(cfg.data_path.split('/')[-1])

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

try:

	saver.restore(sess, load_model_path)
	print("CNN model is loaded !")

except Exception as e:

	print("Model is not loaded !")



for epoch_idx in range(cfg.epoch):

	total_training_loss = 0
	total_training_accuracy = 0
	total_testing_loss = 0
	total_testing_accuracy = 0

	training_counter = 0
	for idx_start in tqdm(range(0, total_training_data, cfg.batch_size)):

		idx_end = idx_start + cfg.batch_size

		if idx_end >= total_training_data : idx_end = total_training_data - 1


		train_images, train_labels = utils_train(idx_start, idx_end)

		loss_, accuracy_, _ = sess.run([model.loss, model.accuracy, model.train_step], 
										feed_dict={ model.x: train_images,
													model.y: train_labels,
													model.keep_prob: 1 - cfg.dropout_rate
													})

		total_training_loss += loss_
		total_training_accuracy += accuracy_
		training_counter += 1



	testing_counter = 0

	for idx_start in tqdm(range(0, total_testing_data, cfg.batch_size)):

		idx_end = idx_start + cfg.batch_size

		if idx_end >= total_testing_data : idx_end = total_testing_data - 1


		test_images, test_labels = utils_test(idx_start, idx_end)

		loss_, accuracy_ = sess.run([model.loss, model.accuracy], 
										feed_dict={ model.x: test_images,
													model.y: test_labels,
													model.keep_prob: 1 
													})

		total_testing_loss += loss_
		total_testing_accuracy += accuracy_
		testing_counter += 1



	print("\t----Epoch %d---- "%(epoch_idx+1))
	print(tabulate([["Loss", total_training_loss],["Classification Accuracy", total_training_accuracy/training_counter]],
					   headers=["Training Loss/Accuracy", "Value"]))

	
	print("\n")
	print(tabulate([["Loss", total_testing_loss],["Classification Accuracy", total_testing_accuracy/testing_counter]], 
					   headers=["Evaluation Loss/Accuracy", "Value"]))






