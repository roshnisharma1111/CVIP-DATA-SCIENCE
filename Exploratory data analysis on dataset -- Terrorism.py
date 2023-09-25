#!/usr/bin/env python
# coding: utf-8

# # DATA ANALYTICS

# # TASK 1 : Exploratory data analysis on dataset -Terrorism

# In[1]:


#importing the header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


# In[2]:


data=pd.read_csv('GlobalTerrorismDataset.csv',encoding='latin1')
df=pd.DataFrame(data)
print("data imported")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


for i in df.columns:
    print(i,end=',')


# In[6]:


df=df[["iyear","imonth","iday","country_txt","region_txt","provstate","city","latitude","longitude","location","summary",
       "attacktype1_txt","targtype1_txt","gname","motive","weaptype1_txt","nkill","nwound","addnotes"
      ]]


# In[7]:


df


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df['nkill']=df['nkill'].fillna(0)
df['nwound']=df['nwound'].fillna(0)
df['casuality']=df['nkill']+df['nwound']



# In[11]:


df.isnull().sum()


# In[12]:


df.describe()


# In[13]:


df['region_txt'].unique()


# In[14]:


df['country_txt'].value_counts().head(10)


# In[15]:


df['region_txt'].value_counts().head(10)


# In[16]:


df['provstate'].value_counts().head(10)


# In[17]:


df['attacktype1_txt'].value_counts().head(10)


# In[18]:


df['targtype1_txt'].value_counts().head(10)


# In[19]:


df['gname'].value_counts().head(10)


# In[20]:


df['weaptype1_txt'].value_counts().head(10)


# In[21]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('weapon')
sns.countplot(x=df.weaptype1_txt);


# In[22]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('attack by year')
sns.countplot(x=df.iyear);


# In[23]:


plt.figure(figsize=(50,10))
plt.xticks(rotation=90)
plt.title('attack by country')
sns.countplot(x=df.country_txt);


# In[24]:


plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('REGION')
sns.countplot(x=df.region_txt);


# In[25]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
plt.title('target')
sns.countplot(x=df.targtype1_txt);


# In[26]:


plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
plt.title('attack type')
sns.countplot(x=df.attacktype1_txt);


# In[27]:


plt.figure(figsize=(20,5))
plt.title('month')
sns.countplot(x=df.imonth);


# In[28]:


plt.figure(figsize=(10,5))
plt.title('day')
sns.countplot(x=df.iday);


# In[ ]:





# In[ ]:




