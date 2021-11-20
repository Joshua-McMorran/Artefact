import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\TestData.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)


dictValues = {k:[dic[k] for dic in myArray] for k in myArray[0]}

kmeans = KMeans(n_clusters=2)
X = np.column_stack((dictValues["Mean"], dictValues["Kurtosis"]))
print(X)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()
