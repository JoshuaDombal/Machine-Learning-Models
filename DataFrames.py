import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split


df = pd.read_csv('USA_Housing.csv')
print(df.describe())

#sns.pairplot(df)

sns.distplot(df['Price'], color='red')
plt.show()

sns.heatmap(df.corr())
plt.show()

print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
      dtype='object']]

y = df['Price']



