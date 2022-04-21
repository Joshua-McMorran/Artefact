import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
'''
Housing data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\kc_house_data.csv")
data = sns.pairplot(trainingSet, vars =["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","grade","sqft_above","yr_built","lat","long","sqft_living15","sqft_lot15"])

Breast cancer data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BreastCancer_data.csv")

Brain tumor data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv")
data = sns.pairplot(trainingSet, vars =["Standard Deviation", "Variance","ASM", "Entropy","Energy", "Homogeneity","Skewness", "Class"])
'''


dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv")
dataFile.head()
trainingSet = dataFile.iloc[:,:]

sns.set(style="ticks", color_codes=True)

data = sns.pairplot(trainingSet, vars =["Standard Deviation", "Variance","ASM", "Entropy","Energy", "Homogeneity","Skewness"])
plt.show()
