from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\HousingData_cleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["condition"])

# dataClass = np.array(dictValues["condition"])

X = np.column_stack((dictValues["Price"],dictValues["sqft_living"],dictValues["sqft_lot"],dictValues["floors"],dictValues["condition"],
                      dictValues["yr_built"],dictValues["sqft_living15"], dictValues["bedrooms"]))

#X = np.column_stack((dictValues["Variance"],dictValues["Homogeneity"],dictValues["ASM"], dictValues["Standard Deviation"], dictValues["Energy"], dictValues["Entropy"]))

print("This is X",X.shape)

dataLength = len(X) 

X = StandardScaler().fit_transform(X)
for i in range(10):

    allAccuracysDict = dict()
    #min_samples =15
    # Compute DBSCAN
    print(i)
    db = DBSCAN(eps = 4.8, min_samples= 5+i).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
 
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels))
    #print("min smaples = ",db.min_samples)
    

    for j in range (len(X)):
        #Check for matching Kmeans data to CVS data
        if labels[j] == 1 and dataClass[j] == 1:        
            accurateLabel+=1    
        elif labels[j] == 0 and dataClass[j] == 2:
            accurateLabel+=1
        elif labels[j] == 2 and dataClass[j] == 3:        
            accurateLabel+=1    
        elif labels[j] == 3 and dataClass[j] == 4:
            accurateLabel+=1
        elif labels[j] == 3 and dataClass[j] == 5:
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

lowestAccuracy = 0
lowestAccuracyDict = dict()
for key in allAccuracysDict:
    if allAccuracysDict[key] < highestAccuracy:
        lowestAccuracyDict.clear()
        lowestAccuracyDict[key] = allAccuracysDict[key]
        lowestAccuracy = allAccuracysDict[key]
    elif allAccuracysDict[key] == lowestAccuracy:
        lowestAccuracyDict[key] = allAccuracysDict[key]


#writing to file the highest values of current loop
f  = open ("DBSCAN_HighestAccuracy.txt", "w")
f.write("DBSCAN best acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)


f  = open ("DBSCAN_LowestAccuracy.txt", "w")
f.write("DBSCAN lowest acccuray(s) = " + str(lowestAccuracyDict))
print("Lowest accuracy is  = ", lowestAccuracyDict)


f = open ("DBSCAN_AllAccuracy.txt", "w")
f.write("All Accuracys = " + str(allAccuracysDict))
print("All Accuracys = ", allAccuracysDict)
