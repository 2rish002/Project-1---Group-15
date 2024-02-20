#%%
import numpy as np
import pandas as pd
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# %%
import os
print(os.listdir("/Users/ananyagoel/Desktop/DS-3001/"))
# %%
#Loading Business JSON table info dataframe
users = []
with open('/Users/ananyagoel/Desktop/DS-3001/yelp_dataset/yelp_academic_dataset_business.json') as fl:
    for i, line in enumerate(fl):
        users.append(json.loads(line))
        if i+1 >= 100000:
            break
df_b = pd.DataFrame(users)
df_b.head()
#%%
df_b.info()
#%%
x=df_b['stars'].value_counts()
#%%
x=x.sort_index()
print(x)
#%%
#plot of star distribution
plt.figure(figsize=(8,4))
sns.barplot(x=x.index, y=x.values)
plt.title("Star Rating Distribution")
plt.ylabel('# of businesses', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)

#%%
#Plot of City distribution in dataset
x=df_b['city'].value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]
plt.figure(figsize=(16,4))
ax = sns.barplot(x=x.index, y=x.values, alpha=0.8)
plt.title("Which city has the most reviews?")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('City', fontsize=12)

# %%
#Loading Reviews JSON table info dataframe
reviews = []
with open('/Users/ananyagoel/Desktop/DS-3001/yelp_dataset/yelp_academic_dataset_review.json') as fl:
    for i, line in enumerate(fl):
        reviews.append(json.loads(line))
        if i+1 >= 1000000:
            break
df_r = pd.DataFrame(reviews)
df_r.head()

#%%
x1=df_r['city'].value_counts()

#%%
#Categories plot 
x1=x1.sort_values(ascending=False)
x1=x1.iloc[0:20]
plt.figure(figsize=(16,4))
ax1 = sns.barplot(x=x1.index, y=x1.values, alpha=0.8)
plt.title("Which city has the most reviews?")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('City', fontsize=12)
