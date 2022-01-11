import numpy as np
import csv
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")


myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])

X = np.column_stack((dictValues["Standard Deviation"], dictValues["ASM"]))
dataLength = len(X)

centers = [[1,1,1],[5,5,5],[3,10,10]]

X, _ = make_blobs(n_samples = dataLength, centers = centers, cluster_std = 1.5)



ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)

n_clusters_ = len(np.unique(labels))

print("Number of estimated clusters:", n_clusters_)

colors = 10*['r','g','b','c','k','y','m']

print(colors)
print(labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):

    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')


ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()