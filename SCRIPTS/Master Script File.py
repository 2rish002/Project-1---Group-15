#Master Script File 

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

#%%
yelp_data = pd.read_csv('/Users/aidanwiktorowicz/Desktop/Data Science Project/Yelpdata.csv')

#%%
sid = SentimentIntensityAnalyzer()
#Calculating sentiment scores
yelp_data['polarity_scores'] = yelp_data['text'].apply(lambda text: sid.polarity_scores(str(text)))
yelp_data['compound_score']  = yelp_data['polarity_scores'].apply(lambda score_dict: score_dict['compound'])

#%%
#Calculating and Plotting the mean sentiment scores
import plotly.express as px
xs = pd.DataFrame(yelp_data.groupby(['cat'])['compound_score'].mean())

xs=xs.sort_values(by='compound_score', ascending=False)
xs

fig=px.bar(xs,y='compound_score')
fig.show()


#%%
#Calculating mean star ratings
y=pd.DataFrame(yelp_data.groupby(['cat'])['stars'].mean())
y=y.sort_values(by='stars', ascending=False)
y

#%%
#Conducting ANOVA test
from scipy.stats import f_oneway
anovadf = yelp_data.drop(columns=['stars', 'text', 'polarity_scores'], axis=1)
category_means = [anovadf[anovadf['cat'] == cat]['compound_score'] for cat in anovadf['cat']]

# Perform ANOVA test
f_statistic, p_value = f_oneway(*category_means)

# Output results
print("F-statistic:", f_statistic)
print("P-value:", p_value)

#%%
# How many reviews do we have per category?
average_count = anovadf.groupby(['cat']).count()
average_count

#%%
# Average Score Per Category
average_scores = anovadf.groupby(['cat']).mean()
average_scores

#%%
# Restoring Groupby Output into Dataset
test_df = average_scores.reset_index()
test_df