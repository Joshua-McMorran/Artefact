import numpy as np
import csv
from numpy.lib.shape_base import column_stack
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])

X = np.column_stack(( dictValues["Homogeneity"], dictValues["Entropy"]))
#print("This is X\n", X)

dataLength = len(X) 
centers = X
Z, _ = make_blobs(n_samples=dataLength, centers=centers, cluster_std=0.6)
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using 0.2 original value
bandwidth = estimate_bandwidth(Z, quantile=0.09, n_samples=dataLength)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(Z)
print("This is K\n",Z)
labels = ms.labels_
print("This is labels\n", labels)
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#print("number of estimated clusters : %d" % n_clusters_)
plt.figure(1)
plt.clf()

colors = cycle("mbgrcyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(Z[my_members, 0], Z[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=8.5,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()