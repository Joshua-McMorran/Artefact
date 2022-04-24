import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.lib.shape_base import column_stack
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BreastCancer_cleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])

#X = np.column_stack((dictValues["radius_mean"], dictValues["texture_mean"],dictValues["perimeter_mean"], dictValues["area_mean"],
#                   dictValues["radius_worst"],dictValues["texture_worst"],dictValues["perimeter_worst"],dictValues["area_worst"]))

#Breast cancer training dataset
#BreastCancer_cleaned.csv
#X = np.column_stack((dictValues["radius_mean"], dictValues["texture_mean"],dictValues["perimeter_mean"], dictValues["area_mean"],
#                   dictValues["radius_worst"],dictValues["texture_worst"],dictValues["perimeter_worst"],dictValues["area_worst"]))


X = np.column_stack((dictValues["radius_mean"], dictValues["texture_mean"],dictValues["perimeter_mean"], dictValues["area_mean"],
                        dictValues["radius_worst"],dictValues["texture_worst"],dictValues["perimeter_worst"],dictValues["area_worst"]))

dataLength = len(X) 

for i in range(100):



    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using 0.2 original value
    bandwidth = estimate_bandwidth(X, quantile=0.99, n_samples=dataLength)
    ms = MeanShift( bandwidth=bandwidth, bin_seeding=True, max_iter=250)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)


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

print("clusters = ", n_clusters_)

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