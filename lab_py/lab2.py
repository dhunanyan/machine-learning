#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

import time
import pickle


# In[2]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[3]:


data = (np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int)
print(data)


# In[4]:


obj = {'X': mnist["data"], 'y' : mnist["target"].astype(np.uint8)}

y = pd.Series(obj['y'])
y_sorted = y.sort_values(ascending=True)
print(y_sorted)

X = obj['X']
X_sorted = X.reindex(y_sorted.index)
print(X_sorted)

X_train, X_test = X_sorted[:56000], X_sorted[56000:]
y_train, y_test = y_sorted[:56000], y_sorted[56000:]


# In[5]:


print(y_train)


# In[6]:


print(y_test)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[8]:


print(y_train)


# In[9]:


print(y_test)


# In[10]:


start = time.time()
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
print(time.time() - start)


# In[11]:


print(len(np.where(mnist["target"] == '0')[0])) 


# In[12]:


y_train_0 = (y_train == 0)
print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[13]:


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3, n_jobs=-1)
y_acc_train = accuracy_score(y_train_0, y_train_pred)
print(y_acc_train)


# In[14]:


y_test_0 = (y_test == 0)
print(y_test_0)
print(np.unique(y_test_0))
print(len(y_test_0))


# In[15]:


y_test_pred = cross_val_predict(sgd_clf, X_test, y_test_0, cv=3, n_jobs=-1)
y_acc_test = accuracy_score(y_test_0, y_test_pred)
print(y_acc_test)


# In[16]:


y_acc = [y_acc_train, y_acc_test]


# In[17]:


file_acc_name = "sgd_acc.pkl"

open_file = open(file_acc_name, "wb")
pickle.dump(y_acc, open_file)
open_file.close()

open_file = open(file_acc_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[18]:


start = time.time()
val_score_train_0 = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(time.time() - start)


# In[19]:


file_cva_name = "sgd_cva.pkl"

open_file = open(file_cva_name, "wb")
pickle.dump(val_score_train_0, open_file)
open_file.close()

open_file = open(file_cva_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[20]:


sgd_m_clf = SGDClassifier(n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
#zamiast print(sgd_m_clf.predict([mnist["data"][1], mnist["data"][2]]))
print(sgd_m_clf.predict(mnist["data"].head(2)))


# In[21]:


print(cross_val_score(sgd_m_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)


# In[22]:


conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


# In[23]:


file_conf_mx_name = "sgd_cmx.pkl"

open_file = open(file_conf_mx_name, "wb")
pickle.dump(conf_mx, open_file)
open_file.close()

open_file = open(file_conf_mx_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[26]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
f = "norm_conf_mx.png"
plt.savefig(f)
print(f)


# In[ ]:




