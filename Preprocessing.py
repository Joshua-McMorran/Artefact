from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import style
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
X = np.column_stack((dictValues["Standard Deviation"], dictValues["ASM"],dictValues["Energy"], dictValues["Homogeneity"],
                    dictValues["Mean"],dictValues["Dissimilarity"],dictValues["Variance"]))

dataClass = np.array(dictValues["Class"])

#Standardisation, or mean removal and variance scaling
class PreProcessingStd():
    scaler = preprocessing.StandardScaler().fit(X) 
    scaler.mean_
    scaler.scale_

    X_Scaled = scaler.transform(X)
    #Highest accuracy = 92.61%

class preProcessingTR:
    min_max_scaler = preprocessing.MinMaxScaler()
    X_Scaled = min_max_scaler.fit_transform(X)


allAccuracysDict = dict()
for i in range (100):
    kmeans = KMeans(n_clusters=2, n_init=4, random_state=8, max_iter=300)
    kmeans.fit(PreProcessingStd.X_Scaled)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    for j in range (len(X)):
        #Check for matching Kmeans data to CVS data
        if labels[j] == 1 and dataClass[j] == 0:        
            accurateLabel+=1    
        elif labels[j] == 0 and dataClass[j] == 1:
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


    #Adding into the dictornary
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
f  = open ("KmeansHighestAccuracyPreProcessed.txt", "w")
f.write("Kmeans best acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)

print("\n")

f  = open ("KmeansLowestAccuracyPreProcessed.txt", "w")
f.write("Kmeans lowest acccuray(s) = " + str(lowestAccuracyDict))
print("Lowest accuracy is  = ", lowestAccuracyDict)

print("\n")

f = open ("KmeansAllAccuracysPreProcessed.txt", "w")
f.write("All Accuracys = " + str(allAccuracysDict))
print("All Accuracys = ", allAccuracysDict)
