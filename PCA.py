import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import load_breast_cancer

pd.set_option('display.max_columns', 1000)

cancer = load_breast_cancer()

print(type(cancer))
print(cancer.keys())
print(cancer['DESCR'])


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())

from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
sns.heatmap(df_comp, cmap='plasma')
plt.show()


