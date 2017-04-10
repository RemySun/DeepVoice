print("Importing modules...")

import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
import os
import shelve


print("Loading shelve...")
d = shelve.open("infos_pairs")

print("Loading supervectors...")
input = pickle.load(open("supervectors", "rb"))

print("Generating neural network...")

# Autoencoder with 3 hidden layer
n_input = len(input[0])

print(">>>>>>>>>>>>>>>>>>>" + str(n_input))



n_layer1 = 1000
#n_layer2 = 500
n_hidden = 50

x = tf.placeholder("float", [None, n_input])

# Weights and biases to first layer
Wl1 = tf.Variable(tf.random_uniform((n_input, n_layer1), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bl1 = tf.Variable(tf.zeros([n_layer1]))
l1 = tf.nn.tanh(tf.matmul(x,Wl1) + bl1)
l1_drop = tf.nn.dropout(l1, keep_prob=0.90)

# Weights and biases to second layer

#Wl2 = tf.Variable(tf.random_uniform((n_layer1, n_layer2), -1.0 / math.sqrt(n_layer1), 1.0 / math.sqrt(n_layer1)))
#bl2 = tf.Variable(tf.zeros([n_layer2]))
#l2 = tf.nn.tanh(tf.matmul(l1_drop,Wl2) + bl2)
#l2_drop = tf.nn.dropout(l2, keep_prob=0.90)

# Weights and biases to hidden layer

Wh = tf.Variable(tf.random_uniform((n_layer1, n_hidden), -1.0 / math.sqrt(n_layer1), 1.0 / math.sqrt(n_layer1)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(l1_drop,Wh) + bh)
h_drop = tf.nn.dropout(h, keep_prob=0.90)

# Weights and biases to fifth layer tied to hidden

Wl4 = tf.Variable(tf.random_uniform((n_hidden, n_layer1), -1.0 / math.sqrt(n_hidden), 1.0 / math.sqrt(n_hidden)))
bl4 = tf.Variable(tf.zeros([n_layer1]))
l4 = tf.nn.tanh(tf.matmul(h_drop,Wl4) + bl4)
l4_drop = tf.nn.dropout(l4, keep_prob=0.90)

# Weights and biases to seventh layer tied to first

#Wl5 = tf.transpose(Wl2)
#bl5 = tf.Variable(tf.zeros([n_layer1]))
#l5 = tf.nn.tanh(tf.matmul(l4_drop,Wl5) + bl5)
#l5_drop = tf.nn.dropout(l5, keep_prob=0.90)

# Weights and biases to output layer
Wo = tf.Variable(tf.random_uniform((n_layer1, n_input), -1.0 / math.sqrt(n_layer1), 1.0 / math.sqrt(n_layer1)))
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.matmul(l4_drop,Wo) + bo)

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
path = os.getcwd() + "/model.ckpt"
saver.restore(sess, path)



print("Computing Deepvectors...")
deep_vectors = sess.run(h, feed_dict={x: input})

def norm(a):
    d = 0
    for i in a:
        d += i**2
    return np.sqrt(d)

def dist(a,b):
    l = [i-j for i,j in zip(a,b)]
    return norm(l)

def cosine(a,b):
    dot = [i*j for i,j in zip(a,b)]
    return abs(sum([d/(norm(a)*norm(b)) for d in dot]))

index = d['index']
nb_pairs = index.shape[0]
print("Computing distance for the pairs...")
distances = [dist(deep_vectors[index[i,0]], deep_vectors[index[i,1]]) for i in range(nb_pairs)]
distances_cos = [cosine(deep_vectors[index[i,0]], deep_vectors[index[i,1]]) for i in range(nb_pairs)]


fscores = open("deep.scores", "w")
fnames = d["fnames"]
speakers = np.array(d["speakers"])
print("Printing output to 'deep.scores'...")
for i in range(nb_pairs):
    if i%100 == 0:
        print(str(i) + "/" + str(nb_pairs))
    fscores.write(speakers[i, 0] + " ")
    fscores.write(fnames[i, 0] + " ")
    fscores.write(speakers[i, 1] + " ")
    fscores.write(fnames[i, 1][:-1] + " ")
    fscores.write(str(distances[i]) + "\n")

fscores.close();

print("Done mon pote !")
