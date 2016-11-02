import tensorflow as tf
import numpy as np
import math
import random
#import pandas as pd
#import sys

A = [[random.randint(0, 10) for j in range(1)] for i in range(1000)]
B = [[42] for a in A]

print(A)
print(B)

input = np.array(A)

# Code here for importing data from file

output = B
noisy_input = input


input_data = input
output_data = output

# Autoencoder with 1 hidden layer
n_samp, n_input = input_data.shape 
n_hidden = 20

x = tf.placeholder("float", [None, n_input])
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
h_drop = tf.nn.dropout(h, keep_prob=0.99)
# Weights and biases to hidden layer


Wh2 = tf.Variable(tf.random_uniform((n_hidden, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh2 = tf.Variable(tf.zeros([n_hidden]))
h2 = tf.nn.sigmoid(tf.matmul(h_drop,Wh2) + bh2)
# Weights and biases to hidden layer

Wo = tf.Variable(tf.random_uniform((n_hidden, 1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bo = tf.Variable(tf.zeros([1]))
y = tf.nn.relu(tf.matmul(h2,Wo) + bo)
# Objective functions
y_ = tf.placeholder("float", [None,1])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 1000
batch_size = min(50, n_samp)

for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data
    batch_ys = output_data
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})

print "Target:"
print output_data
print "Final activations:"
print sess.run(y, feed_dict={x: input_data})
print "Final weights (input => hidden layer)"
print sess.run(Wh)
print "Final biases (input => hidden layer)"
print sess.run(bh)
print "Final biases (hidden layer => output)"
print sess.run(bo)
print "Final activations of hidden layer"
print sess.run(h, feed_dict={x: input_data})


print np.mean([b[0] for b in B])



print "Test:"
print sess.run(y, feed_dict={x: [[10]]})
print sess.run(y, feed_dict={x: [[20]]})
print sess.run(y, feed_dict={x: [[30]]})