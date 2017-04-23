__author__ = 'Donnie Kim and Chao Zhou'

import numpy as np
import matplotlib.pyplot as plt
import time,os
import tensorflow as tf
import pickle
import pandas as pd
import scipy.misc as smp
import random
import time
from collections import Counter
from sklearn.cross_validation import train_test_split

# Train code start

# MAKE SURE YOU CHANGE THIS NUMBER
batch_size = 64
num_epoch = 13
# num_data = 70000 * 5    # train
num_data = 520000 # extra cropped. 28000 for validation

# training_file_name. Must check num_data with the file to ensure the dimension match
training_file_name = ["cropped_extra_upperleft", "cropped_extra_lowerleft", "cropped_extra_upperright", "cropped_extra_lowerright", "cropped_extra_middle"]
model_name = "ChaoZhou_DonnieKim_SVHN_CNN_Ensemble_model"

# helper methods for creating CNN
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(var,name):
    """
    Attach summaries to Tensor
    :param var: variable that is in interest
    :param name: name of the variable that one is interested in. must be passed as a string.
    """
    # within the scope of summaries
    with tf.name_scope('summaries'):

        mean = tf.reduce_mean(var)  # obtain mean of the given variable
        tf.summary.scalar('mean/' + name, mean) # get mean with name of mean/name of var

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean))) # evaluate the std of the given variable
        tf.summary.scalar('stddev/' + name, stddev)  # get min with name of stddev/name of var
        tf.summary.scalar('max/' + name, tf.reduce_max(var)) # get max with name of max/name of var
        tf.summary.scalar('min/' + name, tf.reduce_min(var)) # get min with name of min/name of var
        tf.summary.histogram(name,var)  # get histogram of var


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():

    graph = tf.Graph()

    with graph.as_default():
        
        # placeholders for input data batch_size x 32 x 32 x 3 and labels batch_size x 10
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # defining decaying learning rate
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-4, global_step=global_step, decay_steps=10000, decay_rate=0.9)

        # Conv Layer 1: with 32 filters of size 5 x 5 
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(x, [-1, 28, 28, 3])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Conv Layer 2: with 32 filters of size 5 x 5
        W_conv2 = weight_variable([5, 5, 32, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        # Pool Layer 1
        # 14 * 14
        h_pool1 = max_pool_2x2(h_conv2)

        # Conv Layer 4: with 64 filters of size 5 x 5 
        W_conv3 = weight_variable([5, 5, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
        
        # Conv Layer 5: with 64 filters of size 5 x 5
        W_conv4 = weight_variable([5, 5, 64, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

        # Pool Layer 2
        # 7 * 7
        h_pool2 = max_pool_2x2(h_conv4)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        
        # flatening output of pool layer to feed in FC layer
        h_pool4_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        
        # FC layer1
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Output
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        
        # step size
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step)
        
        # evaluate how many times the model got right
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        
        # evaludate accuracy 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # save our trained model
        saver = tf.train.Saver()

        # Add a scalar sumary for the snapshot of accuracy and loss
        accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)
        loss_summary = tf.summary.scalar(name= 'loss', tensor = cross_entropy)

        # Add statistics summary for the hidden layers
        # Activation# means Relu non-linearity on Conv layer#
        actfun1_summary = variable_summaries(var=h_conv1, name='Activation1')
        actfun2_summary = variable_summaries(var=h_conv2, name='Activation2')
        actfun3_summary = variable_summaries(var=h_conv3, name='Activation3')
        actfun4_summary = variable_summaries(var=h_conv4, name='Activation4')
        # actfun5_summary = variable_summaries(var=h_conv5, name='Activation5')
        # actfun6_summary = variable_summaries(var=h_conv6, name='Activation6')

        # Build a summary operation based onf the TF collection of summaries
        summary_merged = tf.summary.merge_all()

    num_steps = int(num_data / batch_size) * num_epoch 
    for index in range(5):
        with tf.Session(graph=graph) as session:
            print('Variables Initialized')
            init = tf.global_variables_initializer()
            session.run(init)
            # MAKE SURE YOU CHANGE THE SUBFOLDER EVERYTIME TO SAVE THE GRAPHS AND SUMMARIES!!!
            result_dir = 'C:/Users/Donnie/Documents/Comp540_project/result/' + model_name + training_file_name[index]
            training_dir = result_dir + '/training'
            validation_dir = result_dir + '/validation'

            print("Loading training data")
            # for memory efficiency, specify the dtype for each column Note this is specific to cropped image
            dictionary = {'label': int}
            tag = 'pixel'
            for i in range(3072):
                tag_num = tag + str(i)
                dictionary[tag_num] = int 
            # Read X and y from the dataset (train and test)
            # pickle_name = training_file_name[index] + ".p"
            # data_train = pickle.load(open(pickle_name, "rb"))
            csv_name = training_file_name[index] + ".csv"
            print(csv_name)
            data_train = pd.read_csv(csv_name, dtype = dictionary)
            # pickle.dump(data_train, open(pickle_name, "wb"), protocol=4)
            print("Finished loading")
            train_data = data_train[data_train.columns[1:]].values
            train_data = train_data.reshape(28, 28, 3, -1)
            train_labels = np.array([data_train.label.values]).T
            # first we will normalize image data in range of -1 and 1.
            train_data = train_data.astype('float32') / 128.0 - 1
            # reshaping np array so that we can access data in CNN friendly format i.e. [i,:,:,:] from [:,:,:,i]
            train_data = np.transpose(train_data, (3, 0, 1, 2))
            #chaning class labels range 1-10 to 0-9
            train_labels[train_labels == 10] = 0
            # processing labels in CNN friendly format i.e. 1-hot-encoding
            num_labels = 10
            train_labels = train_labels[:,0]
            train_labels = (np.arange(num_labels) == train_labels[:, None]).astype(np.float32)
            # splitting data in training and validation sets old school way
            train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, 
                                                                                  train_size=num_data)
            
            train_summary_writer = tf.summary.FileWriter(training_dir, session.graph)
            val_summary_writer = tf.summary.FileWriter(validation_dir)

            for i in range(num_steps):
                offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_data[offset:(offset + batch_size), :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 0.5}

                # Train
                train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

                # For every 1000th step, record the summary
                if i%1000 == 0:
                # For every epoch, record the summary

                    # Obtain train and validation summary
                    summary_train, acc_train = session.run([summary_merged,accuracy], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 1.0})
                    train_summary_writer.add_summary(summary_train,i)
                    # train_summary_writer.flush()

                    summary_val, acc_val = session.run([summary_merged,accuracy], feed_dict={x:valid_data, y_: valid_labels, keep_prob: 1.0})
                    val_summary_writer.add_summary(summary_val,i)
                    # val_summary_writer.flush()
                    # val_summary_writer.get_logdir()

                    # Print out the accuracy for every epoch
                    print("iteration: %d, training accuracy: %g, validation accuracy: %g" % 
                          (i ,acc_train, acc_val ))

            train_data = None 
            data_train = None 
            print('saving trained model')
            saved_model_path = "saved_models/" + model_name + training_file_name[index] + ".ckpt"
            save_path = saver.save(session, saved_model_path)
            print('model saved!')
            session.close()
            print('session closed!') 


if __name__ == "__main__":
    main()


