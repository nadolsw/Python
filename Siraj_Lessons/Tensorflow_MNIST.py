#Based on Siraj's 'Tensorflow in 5 Minutes' youtube video: https://www.youtube.com/watch?v=2FmcHiLCwTU&vl=en #
#Uses TF to build neural network that classifies hand written digits from MNIST dataset#
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST data", one_hot=True)

#Set hyperparameters
learning_rate = 0.01
training_iteration = 50
batch_size = 100
display_step = 1
output_logs = 'C:\\Users\\nadolsw\\Desktop\\Python\\Siraj\\output'

#Initiailize placeholder variables
x = tf.placeholder("float", [None, 784]) #number of pixels per image - 28*28=784 input pixels
y = tf.placeholder("float", [None, 10]) #10 output classes - numbers 0-9

#Set model weights & biases
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Scope #1 - define model
with tf.name_scope("wx_b") as scope:
	#construct linear model
	model = tf.nn.softmax(tf.matmul(x, w) + b)

#Summary operations to collect data
weight_hist = tf.summary.histogram("weights", w)
bias_hist = tf.summary.histogram("biases", b)

#Sope #2 - minimize error using cross-entropy
with tf.name_scope("cost_function") as scope:
	cost_function = -tf.reduce_sum(y*tf.log(model))
	tf.summary.scalar("cost_function", cost_function)

#Scope #3 - gradient descent
with tf.name_scope("train") as scope:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#Initialize variables
init = tf.global_variables_initializer()

#Merge summaries into single operator
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)

	#Set log writer to folder /tmp/tensorflow_logs
	#filepath='C:/Users/nadolsw/Desktop/Python/Udacity/Intro to Deep Learning'
	summary_writer = tf.summary.FileWriter(output_logs, sess.graph)

	#Training cycle
	for iteration in range(training_iteration):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		#loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			#fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			#compute avg loss
			avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			#write log for each iteration
			summary_str = sess.run(merge_summary, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, iteration*total_batch + i)
		#display logs each iteration
		if iteration % display_step == 0:
			print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:9f}".format(avg_cost))

		print ("Tuning Complete!")

		#Test model
		predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
		#Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
		print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

#To visualize tensorboard:
###1. open windows command terminal
###2. cd to output directory (C:\Users\nadolsw\Desktop\Python\Siraj\output)
###3. submit tensorboard --logdir=C:\Users\nadolsw\Desktop\Python\Siraj\output --port 6006
###4. navigate to http://desktop-ddi6ig4:6006/#scalars in web browser
