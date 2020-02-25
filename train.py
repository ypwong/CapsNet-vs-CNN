from conf import args as cfg
import numpy as  np 
import os
import utils
import CNN
import CapsNet
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib
matplotlib.pyplot.switch_backend('agg')

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 

tf.reset_default_graph()
sess = tf.InteractiveSession()

if cfg.model == 'CNN':

	model = CNN.CNN(image_height=cfg.image_height,
					image_width=cfg.image_width,
					image_depth= cfg.image_depth,
					learning_rate=cfg.learning_rate,
					decay_steps= cfg.decay_steps,
					decay_rate = cfg.decay_rate,
					freeze_conv = cfg.freeze_conv
					)


if cfg.model == 'CapsNet':

	model = CapsNet.CapsNet(image_height=cfg.image_height,
							image_width=cfg.image_width,
							image_depth= cfg.image_depth,
							learning_rate=cfg.learning_rate,
							decay_steps= cfg.decay_steps,
							decay_rate = cfg.decay_rate,
							primary_caps_vlength=cfg.primary_caps_vlength  ,
							digit_caps_vlength=cfg.digit_caps_vlength ,
							epsilon=cfg.epsilon,
							lambda_=cfg.lambda_,
							m_plus=cfg.m_plus,
							m_minus=cfg.m_minus,
							reg_scale=cfg.reg_scale,
							routing_iteration=cfg.routing_iteration,
							freeze_conv = cfg.freeze_conv
							)


utils_train = utils.Utils(data_path=cfg.train_data_path,
					image_height=cfg.image_height,
					image_width=cfg.image_width,
					image_depth=cfg.image_depth)

utils_test = utils.Utils(data_path= cfg.test_data_path_base + str(cfg.train_data_path.split('/')[-2]) + '/',
						image_height=cfg.image_height,
						image_width=cfg.image_width,
						image_depth=cfg.image_depth)

total_training_data = utils_train.total_data
total_testing_data  = utils_test.total_data

save_model_path = './model_ckpt/'+cfg.model+'/'+str(cfg.train_data_path.split('/')[-3]) + '/' +str(cfg.train_data_path.split('/')[-2]) + '/model.ckpt'
load_model_path = None

if not os.path.exists(save_model_path) : os.makedirs(save_model_path)
#Path to save figures
fig_save_path = cfg.fig_save_path + cfg.train_data_path.split('/')[-2] + '/'+ cfg.train_data_path.split('/')[-3] + '/' + cfg.model + '/'
if not os.path.exists(fig_save_path) : os.makedirs(fig_save_path)


if cfg.freeze_conv:

	load_model_path = './model_ckpt/'+cfg.model+'/'+str(cfg.train_data_path.split('/')[-3] + '/1_polygon/model.ckpt')

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

try:

	saver.restore(sess, load_model_path)
	print("Prevous model is loaded !")

except Exception as e:

	print("Model is not loaded !")


pred_list   = []
actual_list = []
training_losses = []
training_accuracies = []
testing_losses = []
testing_accuracies = []

best_test_accuracy = 0

for epoch_idx in range(cfg.epoch):

	total_training_loss = 0
	total_training_accuracy = 0
	total_testing_loss = 0
	total_testing_accuracy = 0

	training_counter = 0
	for idx_start in tqdm(range(0, total_training_data, cfg.batch_size)):

		idx_end = idx_start + cfg.batch_size

		train_images, train_labels = utils_train(idx_start, idx_end)

		loss_, accuracy_, _ = sess.run([model.loss, model.accuracy, model.train_step], 
										feed_dict={ model.x: train_images,
													model.y: train_labels,
													model.keep_prob: 1 - cfg.dropout_rate
													})

		total_training_loss += loss_
		total_training_accuracy += accuracy_
		training_counter += 1

	#append the training losses and accuracies
	training_losses.append(total_training_loss)
	training_accuracies.append(total_training_accuracy/training_counter)

	curr_lr = sess.run(model.decayed_lr)
	print("Current learning rate : " , curr_lr)


	testing_counter = 0

	for idx_start in tqdm(range(0, total_testing_data, cfg.batch_size)):

		idx_end = idx_start + cfg.batch_size

		test_images, test_labels = utils_test(idx_start, idx_end)

		loss_, accuracy_ = sess.run([model.loss, model.accuracy], 
										feed_dict={ model.x: test_images,
													model.y: test_labels,
													model.keep_prob: 1 
													})

		total_testing_loss += loss_
		total_testing_accuracy += accuracy_
		testing_counter += 1

	#save the model with the highest accuracy
	if total_testing_accuracy/testing_counter > best_test_accuracy:
		saver.save(sess, save_model_path)
		best_test_accuracy = total_testing_accuracy/testing_counter

	#append the testing losses and accuracies
	testing_losses.append(total_testing_loss)
	testing_accuracies.append(total_testing_accuracy/testing_counter)

	if epoch_idx == cfg.epoch - 1:

		for idx_start in tqdm(range(0, total_testing_data, cfg.batch_size)):

			idx_end = idx_start + cfg.batch_size

			test_images, test_labels = utils_test(idx_start, idx_end)

			predicted_label = sess.run([model.softmaxed_prediction], feed_dict={
																				model.x: test_images,
																				model.y: test_labels,
																				model.keep_prob: 1 
																				})


			target_labels = [utils_test.inverse_classes_dict[utils_test.mode][int(x)] for x in np.argmax(test_labels, axis=1)]
			pred_labels   = [utils_test.inverse_classes_dict[utils_test.mode][int(x)] for x in np.argmax(predicted_label[0], axis=1)]

			pred_list.append(pred_labels)
			actual_list.append(target_labels)





	print("\t----Epoch %d---- "%(epoch_idx+1))
	print(tabulate([["Loss", total_training_loss],["Classification Accuracy", total_training_accuracy/training_counter]],
					   headers=["Training Loss/Accuracy", "Value"]))

	
	print("\n")
	print(tabulate([["Loss", total_testing_loss],["Classification Accuracy", total_testing_accuracy/testing_counter]], 
					   headers=["Evaluation Loss/Accuracy", "Value"]))


#After the training and testing
print("Best test accuracy : ", best_test_accuracy)


pred_list = sum(pred_list, [])
actual_list = sum(actual_list, [])

data = {'predicted': pred_list,
        'actual':   actual_list,
        }

df = pd.DataFrame(data, columns=['actual','predicted'])

confusion_matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
plt.savefig(fig_save_path+'confusion_matrix.png')

actual_list = [utils_test.classes_dict[utils_test.mode][x] for x in actual_list]
pred_list = [utils_test.classes_dict[utils_test.mode][x] for x in pred_list]


precision = round(precision_score(actual_list, pred_list),3)
recall = round(recall_score(actual_list, pred_list),3)
f1_score = round(f1_score(actual_list, pred_list),3)

info = "Precision : " + str(precision) +" Recall : " + str(recall) + " F1 Score : " + str(f1_score) + " Best Test Accuracy : " + str(best_test_accuracy)

stats_file = open(fig_save_path + 'stats.txt', 'w')
stats_file.write(info)
plt.clf()

x_axis = [x for x in range(cfg.epoch)] #number of epochs

y_axis_training_loss = training_losses
y_axis_training_acc  = training_accuracies
y_axis_testing_loss	 = testing_losses
y_axis_testing_acc 	 = testing_accuracies

plt.plot(x_axis, y_axis_training_loss, 'r--', label='Training Loss')
plt.plot(x_axis, y_axis_testing_loss, 'b--', label='Testing Loss')
plt.title("Training and Testing Losses")
plt.xlabel("Epoch")
plt.ylabel("Accuracies")
plt.legend()




plt.savefig(fig_save_path+'loss_result.png')

plt.clf()

plt.plot(x_axis, y_axis_training_acc, 'r--', label='Training Accuracies')
plt.plot(x_axis, y_axis_testing_acc, 'b--', label='Testing Accuracies')
plt.title("Training and Testing Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracies")
plt.legend()

plt.savefig(fig_save_path+'accuracy_result.png')

