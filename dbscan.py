import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)


dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])


X = np.column_stack((dictValues["Variance"],dictValues["Homogeneity"],dictValues["ASM"], dictValues["Standard Deviation"], dictValues["Energy"], dictValues["Entropy"]))

print("This is X",X.shape)

dataLength = len(X) 

X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db = DBSCAN(eps=0.8, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(labels)

print("number of estimated clusters : %d" % n_clusters_)