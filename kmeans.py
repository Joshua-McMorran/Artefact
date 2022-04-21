from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

#CSV to Dictonary
myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["Class"])
#print("dic", dictValues)
#X = np.column_stack((dictValues["Standard Deviation"], dictValues["ASM"],dictValues["Energy"], dictValues["Homogeneity"],
#                    dictValues["Mean"],dictValues["Dissimilarity"],dictValues["Variance"]))

X = np.column_stack((dictValues["Variance"],dictValues["Homogeneity"],dictValues["ASM"], dictValues["Standard Deviation"], dictValues["Energy"], dictValues["Entropy"]))

dataLength = len(X)
print("actual label count = ", dataLength)

#Creating dictonary to store all accuracy of hyperparameter tuning
allAccuracysDict = dict()


for i in range (50):

    kmeans = KMeans(n_clusters=2, random_state = i, n_init = i+1, tol=i)
    kmeans.fit(X)
    #print("random state = ", kmeans.random_state)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
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

#writing to file the highest values of current loop
f  = open ("KmeansAccuracy.txt", "w")
f.write("Kmeans best acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)      




# colors = ("mbgrcyk")
# for i in range(len(X)):
#    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10) 

# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=180, linewidths = 5.2, zorder = 12)
# plt.show()
