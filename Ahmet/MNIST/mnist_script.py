# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Created based on the following tutorial: http://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-tensorflow/
"""

#%% IMPORT NECESSARY PACKAGES

# To load the MNIST dataset you will need to install 'python-mnist'
# Install it with 'pip install python-mnist'
#pip install python-mnist
#pip install utils

import sys
sys.path.insert(0,'..')

import numpy as np
import tensorflow as tf
import mnist

#from cnn_models.lenet5 import *
#from cnn_models.lenet5_like import *
#from cnn_models.alexnet import *
#from cnn_models.vggnet16 import *

from utils import *
#import load_data as ld
from collections import defaultdict

from tensorflow.examples.tutorials.mnist import input_data
mndata = input_data.read_data_sets("MNIST data", one_hot=True)

#%% TEST CODE TO ENSURE TF IS WORKING AS INTENDED

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2,2], tf.float32))
    
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(session.run(a))
    print(session.run(b))

#%% LOAD DATA
    
mnist_folder = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/MNIST/'
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10

#mndata = MNIST(mnist_folder)
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()

mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)

print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_size*mnist_image_size*1))
print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))

print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)

train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels