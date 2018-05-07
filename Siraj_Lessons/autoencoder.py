import tensorflow as tf
import numpy as np
#import tensorflow.examples.tutorials.mnist.input_data as input_data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data", one_hot=True)