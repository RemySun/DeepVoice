import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import sys
import pickle

print("Loading data...")

data = pickle.load(open("valid_data.p", "rb"))
labels = pickle.load(open("valid_labels.p", "rb"))

print("Generating neural network...")

# Autoencoder with 3 hidden layer
n_input = len(data[0])

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

print("Loading trained model...")
saver.restore(sess, "model.ckpt")

print("Computing intermediate representations")

input = np.array(data)

deep_vectors = sess.run(h, feed_dict={x: input})

pairs = []

for i in range(len(deep_vectors)-1):
    for j in range(len(deep_vectors)-i-1):
        if labels[i] == labels[i+j+1]:
            pairs.append([deep_vectors[i],deep_vectors[i+j+1]])


print("Computing threshold...")

def norm(a):
    d = 0
    for i in a:
        d += i**2
    return np.sqrt(d)

def cosine(a,b):
    dot = [i*j for i,j in zip(a,b)]
    return sum([d/(norm(a)*norm(b)) for d in dot])

dist_same = [cosine(p[0],p[1]) for p in pairs]

threshold = min(dist_same)

oops = 0
alright = 0

print("Computing results...")

for i in range(len(deep_vectors)):
    for j in range(len(deep_vectors)-i):
        if labels[i] != labels[i+j]:
            tmp = cosine(deep_vectors[i],deep_vectors[i+j])
            if tmp >= threshold:
                oops += 1
            else:
                alright += 1

print("Threshold:")
print(threshold)
print("\nFine:")
print(alright)
print("\nless fine...")
print(oops)

