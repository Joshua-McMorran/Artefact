import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
'''
Housing data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\kc_house_data.csv")
data = sns.pairplot(trainingSet, vars =["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade", "sqft_above", "yr_built", "lat long", "sqft_living15", "sqft_lot15", "condition"])

Breast cancer data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BreastCancer_cleaned.csv.csv")
data = sns.pairplot(trainingSet, vars = ["radius_mean","texture_mean","perimeter_mean","area_mean", "radius_worst", "texture_worst", "perimeter_worst", "area_worst"])

Brain tumor data
dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\BrainTumorCleaned.csv")
data = sns.pairplot(trainingSet, vars =["Standard Deviation", "Variance","ASM", "Entropy","Energy", "Homogeneity","Skewness", "Class"])
'''

dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\HousingData_cleaned.csv")
dataFile.head()
trainingSet = dataFile.iloc[:,:]


sns.set(style="ticks", color_codes=True)
data = sns.pairplot(trainingSet, vars =["Class", "bathrooms", "sqft_living", "sqft_lot", "floors", "grade", "yr_built", "sqft_living15","condition"])

plt.show()