from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

positiveLable, negativeLable , positiveClass, negativeClass, totalAccuracy = 0,0,0,0,0

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])

X = np.column_stack((dictValues["Standard Deviation"], dictValues["ASM"]))
print(X)
dataLength = len(X)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("Coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10) 
    # if labels && class labels == 1 add
    if (labels[i] == 1): 
        positiveLable+=1
    else:
        negativeLable+=1

#print("\n",dataClass)
for i in range (len(X)):
    if (dataClass[i]==1):
        positiveClass+=1
    else:
        negativeClass+=1

print("Total postive label:", positiveLable, "Total negative label:", negativeLable)
print(positiveClass)
print(negativeClass)
algorithmAccuracy =  positiveLable/positiveClass 
print("The K-means algorithm is this accurate:",algorithmAccuracy,"%")
#print("this is positive class total", positiveClass)

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=180, linewidths = 5.2, zorder = 12)
plt.show()
