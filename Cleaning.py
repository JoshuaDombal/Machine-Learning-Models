import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')

#plt.figure(figsize=(10,7))
#sns.boxplot(x='Pclass', y='Age', data=train)
#plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin', axis=1,inplace=True)
#sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
#plt.show()

sex = pd.get_dummies(train['Sex'],drop_first=True)
#print(sex)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
#print(embark)

train = pd.concat([train,sex,embark],axis=1)

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1,inplace=True)
print(train.head())