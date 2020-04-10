#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


# In[13]:


df = pd.read_csv("movie_dataset.csv")
print (df.columns)


# In[3]:


features = ['keywords','cast','genres','director']


# In[14]:


for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)	

df["combined_features"] = df.apply(combine_features,axis=1)
print ("Combined Features:", df["combined_features"].head())


# In[7]:


cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])


# In[8]:


cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "Avatar"


# In[9]:


movie_index = get_index_from_title(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))


# In[10]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


# In[16]:


i=0
print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
		print (get_title_from_index(element[0]))
		i=i+1
		if i>10:
			break


# In[ ]:





# In[ ]:




