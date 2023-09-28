#!/usr/bin/env python
# coding: utf-8

# # Phase 1 - Breast cancer prediction

# In[82]:


import pandas as pd
import numpy as np
df=pd.read_csv('data.csv')


# In[83]:


df


# In[84]:


df.head()


# In[85]:


df.shape


# In[86]:


df.columns


# # cleaning data

# In[87]:


#missing values
cols_with_missing=[col for col in df.columns
                  if df[col].isnull().any()]


# In[88]:


missing_value=(df.isnull().sum())


# In[89]:


missing_value


# In[90]:


df=df.drop(cols_with_missing,axis=1)


# In[91]:


print(df['diagnosis'].describe())


# In[92]:


print(df.head())


# # Categorical Variables

# In[93]:


diagnosis_mapper={"B":0,"M":1}
df['diagnosis']=df['diagnosis'].replace(diagnosis_mapper)


# In[94]:


df=df.drop("id",axis=1)


# In[95]:


df


# # Modelling

# In[96]:


from sklearn.neighbors import NearestCentroid, KNeighborsClassifier


# In[97]:


def get_score_nearest_centroid():
    scores=cross_val_score(NearestCentroid(),x,y,cv=3,scoring='accuracy')
    return scores.mean()


# In[98]:


def get_score_k_neighbors(n_neighbors):
    scores=cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors),x,y,cv=3,scoring='accuracy')
    return scores.mean()


# In[99]:


from sklearn.ensemble import RandomForestClassifier


# In[100]:


def get_score_random_forest(n_estimators):
    scores=cross_val_score(RandomForestClassifier(n_estimators=n_estimators),x,y,cv=3,scoring='accuracy')
    return scores.mean()


# In[101]:


from sklearn import svm


# In[102]:


def get_score_svm(kernel):
    clf=svm.SVC(kernel=kernel,C=1)
    scores=cross_val_score(clf,x,y,cv=1,scoring='accuracy')
    return scores.mean()


# # Prediction

# In[103]:


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# In[104]:


x=df.copy()
y=df['diagnosis']
x.drop(['diagnosis'],axis=1,inplace=True)


kernels={"linear","poly","rbf","sigmoid"}
results={}


# In[105]:


#adding result for nearest neighbours approaches
results[4]=get_score_nearest_centroid()*100
resultsKNeighbors={}


# In[106]:


print(results)


# In[107]:


import matplotlib.pyplot as plt
classifier_labels=['linear','poly','rbf','sigmoid','NearestCentroid','KNeighbors','RandomForest']
fig, ax=plt.subplots()
ax.bar(list(results.keys()),list(results.values()))
plt.xticks(list(results.keys()),classifier_labels)

#set x ticks and labels
ax.set_ylabel("Accuracy(%)")
ax.set_title('Accuracy by classifier')
fig.set_size_inches(14,8)
plt.show()


# In[ ]:




