import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import sys
import pickle

print("Loading data...")

data = pd.io.parsers.read_csv("data.csv")
data = data.as_matrix()

labels = pickle.load(open("labels.p", "rb"))

print("Preparing data...")

pairs = []

for i in range(data.shape[0]-1):
    for j in range(data.shape[0]-i-1):
        if labels[i] == labels[i+j+1]:
            pairs.append([data[i],data[i+j+1]])

# Code here for importing data from file

input = [p[0] for p in pairs] + [p[1] for p in pairs]

output = [p[1] for p in pairs] + [p[0] for p in pairs]

def norm(a):
    d = 0
    for i in a:
        d += i**2
    d

dist_same = [dist(p[0] - p[1]) for p in pairs]

threshold = max(dist_same)

oops = 0
alright = 0

for i in range(data.shape[0]-1):
    for j in range(data.shape[0]-i-1):
        if labels[i] != labels[i+j+1]:
            tmp = dist(data[i]-data(i+j+1))
            if tmp < threshold:
                oops += 1
            else:
                alright += 1

print("Threshold:")
print(threshold)
print("\nFine:")
print(alright)
print("\nless fine...")
print(oops)

