#!/usr/bin/env python
# coding: utf-8

# Dataset: https://www.kaggle.com/datasets/jainpooja/fake-news-detection

# http://localhost:8888/notebooks/Downloads/Python_dat/fake-news-detection-using-rnn.ipynb
# http://localhost:8888/notebooks/Downloads/Python_dat/fake-news-detection.ipynb
# http://localhost:8888/notebooks/Downloads/Python_dat/fake-news-detection%20(1).ipynb

# In[56]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[57]:


fake=pd.read_csv("C:/Users/palla/Documents/Py files/datasets/Fake_News/Fake.csv")
true=pd.read_csv("C:/Users/palla/Documents/Py files/datasets/Fake_News/True.csv")


# Checking the files

# In[58]:


fake.shape


# In[59]:


fake.head()


# In[60]:


fake.isna().sum()


# In[61]:


true.shape


# In[62]:


true.head()


# In[63]:


true.isna().sum()


# Inserting a new column -- 0 for fake news, and 1 for real news

# In[64]:


fake['type']=0
true['type']=1


# In[65]:


fake.head()
true.head()


# Appending both the true and false news

# In[66]:


totalnews=pd.concat([fake,true])
totalnews


# **Dropping the title, subject and date columns

# In[67]:


totalnews=totalnews.drop(['title', 'subject', 'date'],axis=1)


# In[68]:


totalnews.head()


#  

# Spliting into training and testing datasests

# In[69]:


feature=totalnews['text']
target=totalnews['type']


# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.20, random_state=18)


# In[71]:


#assining the 'text' column to the features variable 
#assining 'type' column to the targets variable
#train_test_split function from scikit-learn is used to split the data into training and testing sets
#We have a test size of 20% and a random state of 18.


# In[72]:


print('X_train')
print(X_train)
print('X_test')
print(X_test)
print('y_train')
print(y_train)
print('y_test')
print(y_test)


# Cleaning our data

# In[73]:


import re 

def clean(data):
    cleaned=[]
    for i in data:
        i=i.lower()
        
        #removing urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        
        
        i = re.sub('\\W', ' ', i) # get rid of non word character and replace with space
        i = re.sub('\n', '', i)   #removes newline characters from the text
        i = re.sub(' +', ' ', i)  #replaces multiple consecutive spaces with a single space
        i = re.sub('^ ', '', i)   #removes leading space at the beginning of text
        i = re.sub(' $', '', i)   #removes trailing space at the end of text
        
        cleaned.append(i)
    return cleaned

X_train = clean(X_train)
X_test = clean(X_test)

