#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import requests
import tarfile
import gzip
import os, sys, tarfile
import shutil


# In[2]:


CSV_URL = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'

req = requests.get(CSV_URL, './data')
url_content = req.content
tgz_file = open('housing.tgz', 'wb')
tgz_file.write(url_content)
tgz_file.close()
    
def extract(tar_url='./data', extract_path='./data'):
    print(tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
try:

    extract(sys.argv[1] + '.tgz')
    print('Done.')
except:
    name = os.path.basename(sys.argv[0])
    print(name[:name.rfind('.')], '<filename>')   

my_tar = tarfile.open('housing.tgz')
my_tar.extract('housing.csv','./data/')
my_tar.close()

with open('./data/housing.csv', 'rt') as f_in:
    with gzip.open('./data/housing.csv.gz', 'wt') as f_out:
        shutil.copyfileobj(f_in, f_out)
        f_out.close()
        f_in.close()
        
os.remove("housing.tgz")


# In[3]:


df = pd.read_csv('./data/housing.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df["ocean_proximity"].describe()


# In[8]:


df["ocean_proximity"].value_counts()


# In[9]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[10]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[11]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[12]:


df.corr()["median_house_value"].sort_values(ascending=False)


# In[13]:


df["median_house_value"]


# In[14]:


pd.Series.reset_index(df.rename(columns={"median_house_value" : "median_house_value (CORR_COLL)"}).corr()["median_house_value (CORR_COLL)"].sort_values(ascending=False), level=None, drop=False, inplace=False).transpose()


# In[15]:


pd.Series.reset_index(df.rename(columns={"median_house_value" : "median_house_value (CORR_COLL)"}).corr()["median_house_value (CORR_COLL)"].sort_values(ascending=False), level=None, drop=False, inplace=False).transpose().to_csv('korelacja.csv', index=False) 


# In[16]:


sns.pairplot(pd.Series.reset_index(df.rename(columns={"median_house_value" : "median_house_value (CORR_COLL)"}).corr()["median_house_value (CORR_COLL)"].sort_values(ascending=False), level=None, drop=False, inplace=False))


# In[17]:


sns.pairplot(df)


# In[18]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set),len(test_set)


# In[19]:


train_set.to_csv('train_set.pkl', index=False)
train_set


# In[20]:


test_set.to_csv('test_set.pkl', index=False)
test_set


# In[21]:


train_set.corr()["median_house_value"].sort_values(ascending=False)


# In[22]:


test_set.corr()["median_house_value"].sort_values(ascending=False)

