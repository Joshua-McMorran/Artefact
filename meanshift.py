import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.lib.shape_base import column_stack
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])

X = np.column_stack((dictValues["Variance"],dictValues["Homogeneity"],dictValues["ASM"], dictValues["Standard Deviation"], dictValues["Energy"], dictValues["Entropy"]))

dataLength = len(X) 
centers = X
Z, _ = make_blobs(n_samples=dataLength, centers=X, cluster_std=0.6)

for i in range(1):
    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using 0.2 original value
    bandwidth = estimate_bandwidth(Z, quantile=0.08, n_samples=dataLength)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Z)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("clusters = ", n_clusters_)

    allAccuracysDict = dict()


    for j in range (len(X)):
        #Check for matching Kmeans data to CVS data
        if labels[j] == 1 and dataClass[j] == 1:        
            accurateLabel+=1   
        elif labels[j] == 0 and dataClass[j] == 0:
            accurateLabel+=1
        elif labels[j]>1:
            falsePositive+=1

        #Sum up CVS
        if (dataClass[j]==1):
            positiveClass+=1
        else:
            negativeClass+=1
        #sum up Kmeans Data
        if (labels[j]==1):
            totalPosLabel+=1
        else:
            totalNegLabel+=1
        

        totalAccuracy = accurateLabel/len(X)
        totalAccuracy = totalAccuracy*100
        totalLabels = totalNegLabel + totalPosLabel

        allAccuracysDict[i] = totalAccuracy

        #clearing values to allow for rerun of loop
        accurateLabel = 0
        positiveClass = 0
        negativeClass = 0 
        totalAccuracy = 0 
        falsePositive = 0
        totalPosLabel = 0
        totalNegLabel = 0
        totalLabels = 0

    #Creates dictornay to store highest vaules
highestAccuracy = 0
highestAccuracyDict = dict()
for key in allAccuracysDict:
    if allAccuracysDict[key] > highestAccuracy:
        highestAccuracyDict.clear()
        highestAccuracyDict[key] = allAccuracysDict[key]
        highestAccuracy = allAccuracysDict[key]
    elif allAccuracysDict[key] == highestAccuracy:
        highestAccuracyDict[key] = allAccuracysDict[key]


f  = open ("MeanShiftAccuracy.txt", "w")
f.write("MeanShift best acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)    
'''
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
'''