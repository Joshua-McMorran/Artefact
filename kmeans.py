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

currentValue = 0

for i in range (10):

    kmeans = KMeans(n_clusters=2, random_state = 3)
    kmeans.fit(X)
    #print("random state = ", kmeans.random_state)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #print("Kmeans lables", labels)
    #print("all labels", dataClass)

    for i in range (len(X)):
        #Check for matching Kmeans data to CVS data
        if labels[i] == 1 and dataClass[i] == 1:        
            accurateLabel+=1    
        elif labels[i] == 0 and dataClass[i] == 0:
            accurateLabel+=1
        elif labels[i]>1:
            falsePositive+=1

        #Sum up CVS
        if (dataClass[i]==1):
            positiveClass+=1
        else:
            negativeClass+=1
        #sum up Kmeans Data
        if (labels[i]==1):
            totalPosLabel+=1
        else:
            totalNegLabel+=1

    totalAccuracy = accurateLabel/len(X)
    totalAccuracy = totalAccuracy*100

    totalLabels = totalNegLabel + totalPosLabel
    # print("Total label = ", totalLabels)
    # print("accurate label = ", accurateLabel)
    # print("Postive Class = ", positiveClass)
    # print("Total accuracy =", totalAccuracy, "%")
    # print("Total false Positive = ", falsePositive)

    allAccuracysDict[kmeans.random_state] = totalAccuracy

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
f.write("Kmeans acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)      




# colors = ("mbgrcyk")
# for i in range(len(X)):
#    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10) 

# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=180, linewidths = 5.2, zorder = 12)
# plt.show()
