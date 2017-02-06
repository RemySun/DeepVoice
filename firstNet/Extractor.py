import re
import glob
import csv
import os
import pickle

Files = glob.glob("data/g256/*.g256")

labels = []

data = []

print("Extracting data...")

nb_files = len(Files)

for i, s in enumerate(Files):
    # Handling name recognition
    labels.append(s.split(".")[1])
    

    print("File "+str(i+1)+"/"+str(nb_files))

    # Creating supervector

    ## storing supervector in tmp file
    bashCommand = 'sgmcopy -x -o "tmp" ' + s
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

train_data = []
train_labels = []
valid_data = []
valid_labels = []

for i in range(len(data)):
    if i % 20 == 0:
        valid_data.append(data[i])
        valid_labels.append(labels[i])
    else:
        train_data.append(data[i])
        train_labels.append(labels[i])

print("Dumping data")

# Write data and labels
t_data = open("train_data.p", "wb")
t_labels = open("train_labels.p", "wb")
v_data = open("valid_data.p", "wb")
v_labels = open("valid_labels.p", "wb")

pickle.dump(train_data, t_data)
pickle.dump(train_labels, t_labels)

pickle.dump(valid_data, v_data)
pickle.dump(valid_labels, v_labels)

t_data.close()
t_labels.close()
v_data.close()
v_labels.close()
