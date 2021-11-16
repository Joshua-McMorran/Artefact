import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataFile = pd.read_csv("D:\\Josh\\UniversityYear3\\Project\\Dissertation and drafts\\Datasets\\TestData.csv")
dataFile.head()
trainingSet = dataFile.iloc[:,:]


sns.set(style="ticks", color_codes=True)
data = sns.pairplot(trainingSet)

plt.show()
