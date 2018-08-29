
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', 1000)

columns_names = ['user_id', 'item_id','rating','timestamp']
df = pd.read_csv('u.data', sep='\t', names=columns_names)

print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')
print(movie_titles.head())

df = pd.merge(df, movie_titles,on='item_id')
print(df.head())

sns.set_style('white')

print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

#ratings['num of ratings'].hist(bins=70)
#plt.show()

#ratings['rating'].hist(bins=70)
#plt.show()

sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=.5)
plt.show()

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
print(moviemat.head())

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1977)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correleation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())









