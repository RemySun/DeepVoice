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

print("Dumping data")

# Write data and labels
f_data = open("data.csv", "w")
f_labels = open("labels.p", "wb")

w_data = csv.writer(f_data)
pickle.dump(labels, f_labels)

w_data.writerows(data)

f_data.close()
f_labels.close()
