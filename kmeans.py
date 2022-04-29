from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import style
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from Preprocessing import PreProcessing as PreP

accurateLabel, positiveClass, negativeClass, totalAccuracy, falsePositive, totalPosLabel, totalNegLabel, totalLabels = 0,0,0,0,0,0,0,0

#CSV to Dictonary
myArray = []
with open('D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\HousingData_cleaned.csv', mode='r') as inp:
    for line in csv.DictReader(inp):
        for pos in line:
            line[pos] = float(line[pos])
        myArray.append(line)

dictValues = {Y:[dic[Y] for dic in myArray] for Y in myArray[0]}
dataClass = np.array(dictValues["condition"])

#Housing datat training dataset
#HousingData_cleaned.csv
#X = np.column_stack((dictValues["bathrooms"],dictValues["sqft_living"],dictValues["sqft_lot"],dictValues["floors"],dictValues["grade"],
#                      dictValues["yr_built"],dictValues["sqft_living15"],dictValues["condition"]))

#        if labels[j] == 1 and dataClass[j] == 1:        
#            accurateLabel+=1    
#        elif labels[j] == 0 and dataClass[j] == 0:
#Used for the brain tumour and breast cancer data due to only needed 2 classifications

#Brain turmour training dataset
#BrainTumorCleaned.csv
#X = np.column_stack((dictValues["Standard Deviation"], dictValues["ASM"],dictValues["Energy"], dictValues["Homogeneity"],
#                    dictValues["Mean"],dictValues["Dissimilarity"],dictValues["Variance"]))
#Breast cancer training dataset
#BreastCancer_cleaned.csv
#X = np.column_stack((dictValues["radius_mean"], dictValues["texture_mean"],dictValues["perimeter_mean"], dictValues["area_mean"],
#                   dictValues["radius_worst"],dictValues["texture_worst"],dictValues["perimeter_worst"],dictValues["area_worst"]))

print(PreP.X_Scaled())

X = np.column_stack((dictValues["Price"],dictValues["sqft_living"],dictValues["sqft_lot"],dictValues["floors"],dictValues["condition"],
                      dictValues["yr_built"],dictValues["sqft_living15"], dictValues["bedrooms"]))

dataLength = len(X)
print("actual label count = ", dataLength)

#Creating dictonary to store all accuracy of hyperparameter tuning
allAccuracysDict = dict()


for i in range (50):
    #(n_clusters=2, random_state = 15, max_iter=275, n_init=20) - brain tumour
    #(n_clusters=2, random_state=80, max_iter=800, n_init= 1) - breast cancer
    #(n_clusters=5, max_iter=336,) - housing data random_state not used due to datapoints being more consistant
    kmeans = KMeans(n_clusters=5, n_init=51-i, max_iter=336+i, random_state= 20+i)
    
    kmeans.fit(X)
    #print("random state = ", kmeans.random_state)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
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

print(kmeans.n_clusters)


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
f  = open ("KmeansHighestAccuracy.txt", "w")
f.write("Kmeans best acccuray(s) = " + str(highestAccuracyDict))
print("Highest accuracy is  = ", highestAccuracyDict)

print("\n")

# f  = open ("KmeansLowestAccuracy.txt", "w")
# f.write("Kmeans lowest acccuray(s) = " + str(lowestAccuracyDict))
# print("Lowest accuracy is  = ", lowestAccuracyDict)

# print("\n")

# f = open ("KmeansAllAccuracys.txt", "w")
# f.write("All Accuracys = " + str(allAccuracysDict))
# print("All Accuracys = ", allAccuracysDict)
