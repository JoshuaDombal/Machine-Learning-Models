import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

print(tips.head())
print(flights.head())
#tc = tips.corr()
#a = flights.pivot_table(index='month',columns='year',values='passengers')
#sns.heatmap(a)
#sns.clustermap(a)
#plt.show()

#sns.distplot(tips['total_bill'], bins = 40)

#sns.jointplot(x='tip',y='total_bill',data=tips, kind='kde')

#sns.pairplot(tips, hue='sex')

#sns.rugplot(tips['total_bill'])

#sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.std)

#sns.countplot(x='sex',data=tips)

#sns.boxplot(x='day',y='total_bill',data=tips)

#sns.violinplot(x='day',y='total_bill',data=tips)

#sns.stripplot(x='day',y='total_bill',data=tips)

#sns.swarmplot(x='day', y='total_bill', data=tips)

#g = sns.FacetGrid(data=tips,col='time',row='smoker')
#g.map(plt.scatter,'tip','total_bill')

# REGRESSION PLOTS _____________________________________________________________________________________

sns.lmplot(x='total_bill',y='tip',data=tips, hue='sex')
plt.show()