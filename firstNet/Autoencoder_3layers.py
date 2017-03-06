import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import sys
import pickle
import sys

sys.stdout = open('training.log', 'w')

print("Loading data...")

#data = pd.io.parsers.read_csv("data.csv")
#data = data.as_matrix()
data = pickle.load(open("train_data.p", "rb"))
labels = pickle.load(open("train_labels.p", "rb"))

print("Preparing data...")

pairs = []

for i in range(len(data)-1):
    for j in range(len(data)-i-1):
        if labels[i] == labels[i+j+1]:
            pairs.append([data[i],data[i+j+1]])

# Code here for importing data from file

input = [p[0] for p in pairs] + [p[1] for p in pairs]

output = [p[1] for p in pairs] + [p[0] for p in pairs]

index = [i for i in range(len(input))]

print("Generating neural network...")

# Autoencoder with 3 hidden layer
n_samp = len(input)
n_input = len(input[0])

n_layer1 = 2000
n_layer2 = 500
n_hidden = 50

x = tf.placeholder("float", [None, n_input])

# Weights and biases to first layer
Wl1 = tf.Variable(tf.random_uniform((n_input, n_layer1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bl1 = tf.Variable(tf.zeros([n_layer1]))
l1 = tf.nn.tanh(tf.matmul(x,Wl1) + bl1)
l1_drop = tf.nn.dropout(l1, keep_prob=0.90)

# Weights and biases to second layer

Wl2 = tf.Variable(tf.random_uniform((n_layer1, n_layer2), -1.0 / math.sqrt(n_layer1), 1.0 / math.sqrt(n_layer1)))
bl2 = tf.Variable(tf.zeros([n_layer2]))
l2 = tf.nn.tanh(tf.matmul(l1_drop,Wl2) + bl2)
l2_drop = tf.nn.dropout(l2, keep_prob=0.90)

# Weights and biases to hidden layer

Wh = tf.Variable(tf.random_uniform((n_layer2, n_hidden), -1.0 / math.sqrt(n_layer2), 1.0 / math.sqrt(n_layer2)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(l2_drop,Wh) + bh)
h_drop = tf.nn.dropout(h, keep_prob=0.90)

# Weights and biases to fifth layer tied to hidden

Wl4 = tf.transpose(Wh)
bl4 = tf.Variable(tf.zeros([n_layer2]))
l4 = tf.nn.tanh(tf.matmul(h_drop,Wl4) + bl4)
l4_drop = tf.nn.dropout(l4, keep_prob=0.90)

# Weights and biases to seventh layer tied to first

Wl5 = tf.transpose(Wl2)
bl5 = tf.Variable(tf.zeros([n_layer1]))
l5 = tf.nn.tanh(tf.matmul(l4_drop,Wl5) + bl5)
l5_drop = tf.nn.dropout(l5, keep_prob=0.90)

# Weights and biases to output layer
Wo = tf.Variable(tf.random_uniform((n_layer1, n_input), -1.0 / math.sqrt(n_layer1), 1.0 / math.sqrt(n_layer1)))
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.matmul(l5_drop,Wo) + bo)

# Objective functions
y_ = tf.placeholder("float", [None,n_input])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#print("Loading trained model...")
#saver.restore(sess, "model.ckpt")

n_rounds = 30
batch_size = min(50, n_samp)

print("Training...")

batch_xs = np.zeros((50,n_input))
batch_ys = np.zeros((50,n_input))

for i in range(n_rounds):
    print("Starting epoch ", i)
    np.random.shuffle(index)
    for j in range(n_samp // batch_size -1):
        for k in range(batch_size):
            batch_xs[k][:] = input[index[j*batch_size + k]]
            batch_ys[k][:] = output[index[j*batch_size + k]]
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
        print("Epoch ", i, " batch ", j," out of ",n_samp//batch_size, " with error ", sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys}))
    # Save the variables to disk.
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)
