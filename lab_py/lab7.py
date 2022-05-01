#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

import pickle


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[3]:


print("\n---------DATA(X)--------")
print(X)


# In[4]:


print("\n-----DATA(y)-----")
print(y)


# In[ ]:


KMeans_lsit=[]
silhouette_list = []

for k in range (8,13):
    print(f"\n\\\\\\\\\\\\\\\\\\\\\\\\\\KMEANS_K={k}//////////////")
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    silhouette_list.append(silhouette_score(X, kmeans.labels_))
    KMeans_lsit.append(y_pred)
    print("------LABELS-----")
    print(kmeans.labels_)
    print("\n-----PREDICT-----")
    print(y_pred)
    print("\n----CLUSTER_CENTERS----")
    print(kmeans.cluster_centers_)
    print("\n---SILHOUETTE---")
    print(silhouette_score(X, kmeans.labels_))


# In[ ]:


file_kmeans_sil_name = "kmeans_sil.pkl"

open_file = open(file_kmeans_sil_name, "wb")
pickle.dump(silhouette_list, open_file)
open_file.close()

open_file = open(file_kmeans_sil_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('\n--------------------------------------------------EXERCICE 2---------------------------------------------')
print(loaded_list)


# In[ ]:


KMeans_10_conf_m = confusion_matrix(y, KMeans_lsit[2])
KMeans_10_conf_m_max_index_list = []
print("\n--------------CONFUSION_MATRIX_K=10---------------------INDEX----MAX")
for i, arr in enumerate(KMeans_10_conf_m):
    KMeans_10_conf_m_max_index_list.append(np.argmax(arr))
    print(arr,"     ", i,"     ", np.argmax(arr))

KMeans_10_conf_m_max_index_list_sorted = np.sort(list(dict.fromkeys(KMeans_10_conf_m_max_index_list)))


# In[ ]:


file_kmeans_argmax_name = "kmeans_argmax.pkl"

open_file = open(file_kmeans_argmax_name, "wb")
pickle.dump(KMeans_10_conf_m_max_index_list_sorted, open_file)
open_file.close()

open_file = open(file_kmeans_argmax_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('\n---EXERCICE 5----')
print(loaded_list)


# In[ ]:


min_dis_list = []

for i, arr in enumerate(X[:300]):
    for j, arr_to_compare in enumerate(X):
        if(j>i):
            dis = np.linalg.norm(X[i] - X[j])
            min_dis_list.append(dis)


# In[ ]:


min_dis_list_sorted = np.sort(min_dis_list)[:10]
file_dist_name = "dist.pkl"

open_file = open(file_dist_name, "wb")
pickle.dump(min_dis_list_sorted, open_file)
open_file.close()

open_file = open(file_dist_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('\n-------------------------EXERCICE 6-------------------------------')
print(loaded_list)


# In[ ]:


s = (min_dis_list_sorted[0] + min_dis_list_sorted[1] + min_dis_list_sorted[2]) / 3
eps_list = []
   
index = s
while(index <= s + 0.1 * s):
    eps_list.append(index)
    index = index + 0.04 * s
    
print('\n-----------------------EPSILON LIST------------------------')
print(eps_list)


# In[ ]:


dbscan_labels_list = []
for eps in eps_list:
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    print(f"\n----DBSCAN_EPS={eps}----")
    print(dbscan.labels_)
    dbscan_labels_list.append(np.sort(list(dict.fromkeys(dbscan.labels_))))


# In[ ]:


dbscan_labels_list_len = []
for i in dbscan_labels_list:
    dbscan_labels_list_len.append(len(i))

file_dbscan_len_name = "dbscan_len.pkl"

open_file = open(file_dbscan_len_name, "wb")
pickle.dump(dbscan_labels_list_len, open_file)
open_file.close()

open_file = open(file_dbscan_len_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('\n-------------------------EXERCICE 8-------------------------------')
print(loaded_list)


# In[ ]:


print('\n---------------------------------------------CHECKING FILES---------------------------------------------')
print('\n---------------------------------------------kmeans_sil.pkl---------------------------------------------')
print(pd.read_pickle("kmeans_sil.pkl"))
print('\n---------------------------------------------kmeans_argmax.pkl--------------------------------------------')
print(pd.read_pickle("kmeans_argmax.pkl"))
print('\n-------------------------------------------------dist.pkl-------------------------------------------------')
print(pd.read_pickle("dist.pkl"))
print('\n----------------------------------------------dbscan_len.pkl----------------------------------------------')
print(pd.read_pickle("dbscan_len.pkl"))


# In[ ]:




