#%%
import numpy as np
import pandas as pd
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %%
import os
print(os.listdir("/Users/ananyagoel/Desktop/DS-3001/"))
# %%
#Business info dataframe
users = []
with open('/Users/ananyagoel/Desktop/DS-3001/yelp_dataset/yelp_academic_dataset_business.json') as fl:
    for i, line in enumerate(fl):
        users.append(json.loads(line))
        if i+1 >= 100000:
            break
df_b = pd.DataFrame(users)
df_b.head()

#%%
#Cleaning dataframe 
df_b=df_b.drop(['name','address','state','postal_code','latitude','longitude','attributes','hours','is_open'],axis=1)
#%%
df_b.info()

#%%
df_b['city'].value_counts()
df=df_b.loc[df_b.city=='Philadelphia']
df.info()

#%%
df['categories'].value_counts()
#%%
df=df.dropna()
df.info()
#%%
filtered_df = df[df['categories'].str.contains('Restaurants')]

filtered_df.info()
#%%
filtered_df['categories'].value_counts
#%%
df_res=filtered_df.drop(['city','stars'],axis=1)
df_res.info()
# %%
#Reviews dataframe
reviews = []
with open('/Users/ananyagoel/Desktop/DS-3001/yelp_dataset/yelp_academic_dataset_review.json') as fl:
    for i, line in enumerate(fl):
        reviews.append(json.loads(line))
        if i+1 >= 1000000:
            break
df_r = pd.DataFrame(reviews)
df_r.head()
# %%
df_r.info()
#%%
df_r=df_r.drop(['user_id','useful','funny','cool','date'],axis=1)

df_r.info()
#%%
new_df=pd.merge(df_res, df_r, on='business_id')


# %%
new_df.info()
# %%
new_df.head()
# %%
new_df['business_id'].value_counts
# %%
new_df=new_df.drop(['business_id'],axis=1)
# %%
new_df=new_df.drop(['review_count'],axis=1)
# %%
new_df=new_df.drop(['review_id'],axis=1)

#%%
new_df['categories'].value_counts()


# %%
#Simplifying categories column into specific cuisines
categories_dict = {"Italian": "Italian","Japanese":"Japanese","Mediterranean":"Mediterranean","American":"American","Korean":"Korean","Indian":"Indian","Thai":"Thai","Chinese":"Chinese","French":"French","Spanish":"Spanish","Mexican":"Mexcian","Southern":"Southern","Carribean":"Carribean","French":"French","Irish":"Irish"}

def get_category(description):
  
  for key, value in categories_dict.items():
    
    if value in description:
      return key
  return None

# %%
new_df['cat']=new_df["categories"].apply(lambda x: get_category(x))
# %%
#Cleaning new dataframe
new_df.info()
# %%
new_df=new_df.drop(['categories'],axis=1)
# %%
new_df.dropna(axis = 0, how = 'any', inplace = True)
# %%
new_df.info()
# %%
#Converting new df into csv for future convinience 
new_df.to_csv('Yelpdata.csv', index=False)

