import re
import glob
import csv
import os
import pickle
import numpy as np
import shelve

Files = glob.glob("data2/*.txt")
ind = open("sanity-1.ndx")

test_files = []

for line in ind:
    [f1, f2] = line.split(" ")
    test_files.append(f1)
    test_files.append(f2)
    
fnames = []

for i, s in enumerate(Files):
    fname = ".".join(s.split("/")[1].split(".")[0:3])
    fnames.append(fname)

non_test_files = []
i=0
for f in fnames:
    i+=1
    print(i)
    if (not (f in test_files)):
        non_test_files.append(f)


labels = []

data = []

print("Extracting data...")

nb_files = len(Files)

for i, s in enumerate(Files):
    if i%100 = 0:
        print(i)
    fname = ".".join(s.split("/")[1].split(".")[0:3])
    if fname in non_test_files:

        # Handling name recognition
        labels.append(s.split(".")[1])
        

        print("File "+str(i+1)+"/"+str(nb_files))

        # Creating supervector

        ## storing supervector in tmp file
        bashCommand = './sgmcopy -x -o "tmp" ' + s
        os.system(bashCommand)

        ## Reading supervector from tmp file
        file = open("tmp",'r')
        lines = file.read().split("\n")
        file.close()
        supervector_t = [re.split(' *',lines[2 + 3*i])[:-1] for i in range((len(lines)-1)//3)]
        supervector_t = [[float(s) for s in l] for l in supervector_t]
        supervector = []
        for l in supervector_t:
            supervector += l

        ## Store supervector
        data.append(supervector)

print("Dumping data")

# Write data and labels
t_data = open("train_data.p", "wb")
t_labels = open("train_labels.p", "wb")

pickle.dump(data, t_data)
pickle.dump(labels, t_labels)

t_data.close()
t_labels.close()
