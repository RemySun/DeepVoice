import re
import glob
import csv
import os
import pickle
import numpy as np
import shelve

Files = glob.glob("data2/*.txt")
ind = open("sanity-1.ndx")
files_name = []
labels = []

print("Extracting data...")

nb_files = len(Files)

pairs = []
speakers = []
same = []
for line in ind:
    [f1, f2] = line.split(" ")
    pairs.append([f1,f2[:-1]])
    speakers.append([f1.split(".")[1], f2.split(".")[1]])
    if (f1.split(".")[1] == f2.split(".")[1]):
        same.append(True)
    else:
        same.append(False)


pairs = np.array(pairs)
data = np.zeros(pairs.shape+(9216,))

for i, s in enumerate(Files):
    if (i%1000 == 999):
        print("File "+str(i+1)+"/"+str(nb_files))
    # Handling name recognition
    fname = ".".join(s.split("/")[1].split(".")[0:3])
    files_name.append(fname)
    
    ## Reading supervector from tmp file
    file = open(s,'r')
    lines = file.read().split("\n")
    file.close()
    supervector_t = [re.split(' *',lines[2 + 3*k])[:-1] for k in range((len(lines)-1)//3)]
    supervector_t = [[float(x) for x in l] for l in supervector_t]
    supervector = []
    for l in supervector_t:
        supervector += l
    ## Stroring supervector in data
    for j in range(data.shape[0]):
        if fname == pairs[j,0]:
            data[j,0,:] = supervector
        elif fname == pairs[j,1]:
            data[j,1,:] = supervector

    

d = shelve.open("pairs")
d['fnames'] = pairs
d['speakers'] = speakers
d['same'] = same
d['supervectors'] = data
d.close()
