import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


#%matplotlib inline

train = pd.read_csv('titanic_train.csv')
#print(train.head())

print(train.columns)

sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
plt.show()

sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Pclass', data=train)
#plt.show()

#sns.distplot(train['Age'].dropna(), kde=False, bins=30)
#train['Age'].plot.hist(bins=35)
#plt.show()

# shows data
#train.info()

#sns.countplot(x='SibSp', data=train)
#plt.show()

#train['Fare'].hist(bins=40,figsize=(10,4))
#plt.show()