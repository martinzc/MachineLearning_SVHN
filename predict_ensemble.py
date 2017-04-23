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
model_name = "svhn_April19_2Layer2Conv_ensemble"

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

    # Test code start

    print("Loading testing data")
    # Read Test data
    # data_test = pickle.load(open("cropped_test.p", "rb"))
    data_test = pd.read_csv('cropped_test.csv')
    # pickle.dump(data_test, open("cropped_test.p", "wb"), protocol=4)
    print("Loaded testing data")
    test_data = data_test.values
    print("Test data read")
    test_data = test_data.reshape(28, 28, 3, -1)
    # normalize the data by  
    test_data = test_data.astype('float32') / 128.0 - 1
    test_data = np.transpose(test_data,(3, 0, 1, 2))

    all_outputs = []
    for index in range(5):
        with tf.Session(graph=graph) as session:
            saved_model_path = "saved_models/" + model_name + training_file_name[index] + ".ckpt"
            saver.restore(session, saved_model_path)
            prediction = tf.argmax(y_conv, 1)
            print("Generating test result ", index+1)
            output_arr = []
            output_arr.append(prediction.eval(feed_dict={x:test_data[:10000], keep_prob: 1.0}, session=session))
            iter_num = (test_data.shape[0] - 10000) // 10000
            for i in range(iter_num):
                current_output = prediction.eval(feed_dict={x:test_data[((i + 1) * 10000) : ((i + 2) * 10000)], keep_prob: 1.0}, session=session)
                output_arr.append(current_output)
            output_arr.append(prediction.eval(feed_dict={x:test_data[(10000 + iter_num * 10000):], keep_prob: 1.0}, session=session))
            test_output = np.concatenate(output_arr)
            test_output[test_output == 0] = 10
            summed_test_output = np.zeros( (test_output.shape[0] // 5), dtype=np.uint8 )
            for i in range(summed_test_output.shape[0]):
                summed_test_output[i] = test_output[i*5 + index]
            all_outputs.append(summed_test_output)
            session.close()
    print("Ensembling the test results")
    for i in range(all_outputs[0].shape[0]):
        out_list = []
        out_list.append(all_outputs[0][i])
        out_list.append(all_outputs[1][i])
        out_list.append(all_outputs[2][i])
        out_list.append(all_outputs[3][i])
        out_list.append(all_outputs[4][i])
        data = Counter(out_list)
        summed_test_output[i] = data.most_common(1)[0][0]
    df = pd.DataFrame(summed_test_output)
    output_csv_path = model_name + ".csv"
    print("Outputting result to", output_csv_path)
    df.to_csv(path_or_buf=output_csv_path, header=['label'])


if __name__ == "__main__":
    main()


