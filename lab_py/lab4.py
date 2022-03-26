#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pickle


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
# print(data_breast_cancer)

data = pd.concat([data_breast_cancer["target"], data_breast_cancer["data"]['mean smoothness'], data_breast_cancer["data"]['mean area']], axis=1)

print(data)


# ## data_iris = datasets.load_iris()
# print(data_iris)

# In[3]:


train, test = train_test_split(data, test_size=.2)


# In[4]:


print(train)


# In[5]:


print(test)


# In[6]:


svm_clf = Pipeline([("linear_svc", LinearSVC(loss="hinge"))])
svm_clf.fit(train[['mean smoothness',  'mean area']], train['target'])


# In[7]:


svm_clf_scl = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge"))])
svm_clf_scl.fit(train[['mean smoothness',  'mean area']], train['target'])


# In[8]:


X_train_pred = svm_clf.predict(train[['mean smoothness',  'mean area']])
svm_clf_train_acc = accuracy_score(train['target'], X_train_pred)
# print(svm_clf_acc)

X_test_pred = svm_clf.predict(test[['mean smoothness',  'mean area']])
svm_clf_test_acc = accuracy_score(test['target'], X_test_pred)
# print(svm_clf_test_acc)


# In[9]:


X_train_pred_scl = svm_clf_scl.predict(train[['mean smoothness',  'mean area']])
svm_clf_scl_train_acc = accuracy_score(train['target'], X_train_pred_scl)
# print(svm_clf_scl_train_acc)

X_test_pred_scl = svm_clf_scl.predict(test[['mean smoothness',  'mean area']])
svm_clf_scl_test_acc = accuracy_score(test['target'], X_test_pred_scl)
# print(svm_clf_scl_test_acc)


# In[10]:


print("train:", svm_clf_train_acc)
print("test:", svm_clf_test_acc)
print("train scaler:", svm_clf_scl_train_acc)
print("test scaler:", svm_clf_scl_test_acc)


# In[11]:


bc_acc = [svm_clf_train_acc, svm_clf_test_acc, svm_clf_scl_train_acc, svm_clf_scl_test_acc]

file_mse_name = "bc_acc.pkl"

open_file = open(file_mse_name, "wb")
pickle.dump(bc_acc, open_file)
open_file.close()

open_file = open(file_mse_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[12]:


iris = datasets.load_iris(as_frame=True)

data_iris = pd.concat([iris["target"]==2, iris["data"]['petal length (cm)'], iris["data"]['petal width (cm)']], axis=1)

print(data_iris)


# In[13]:


iris_train, iris_test = train_test_split(data_iris, test_size=.2)


# In[14]:


print(iris_train)


# In[15]:


print(iris_test)


# In[16]:


iris_svm_clf = Pipeline([("linear_svc", LinearSVC(loss="hinge"))])
iris_svm_clf.fit(iris_train[['petal length (cm)',  'petal width (cm)']], iris_train['target'])


# In[17]:


iris_svm_clf_scl = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge"))])
iris_svm_clf_scl.fit(iris_test[['petal length (cm)',  'petal width (cm)']], iris_test['target'])


# In[18]:


iris_X_train_pred = svm_clf.predict(iris_train[['petal length (cm)',  'petal width (cm)']])
iris_svm_clf_train_acc = accuracy_score(iris_train['target'], iris_X_train_pred)
# print(svm_clf_acc)

iris_X_test_pred = svm_clf.predict(iris_test[['petal length (cm)',  'petal width (cm)']])
iris_svm_clf_test_acc = accuracy_score(iris_test['target'], iris_X_test_pred)
# print(svm_clf_test_acc)


# In[19]:


iris_X_train_pred = svm_clf_scl.predict(iris_train[['petal length (cm)',  'petal width (cm)']])
iris_svm_clf_scl_train_acc = accuracy_score(iris_train['target'], iris_X_train_pred)
# print(svm_clf_acc)

iris_X_test_pred = svm_clf_scl.predict(iris_test[['petal length (cm)',  'petal width (cm)']])
iris_svm_clf_scl_test_acc = accuracy_score(iris_test['target'], iris_X_test_pred)
# print(svm_clf_test_acc)


# In[20]:


print("train:", iris_svm_clf_train_acc)
print("test:", iris_svm_clf_test_acc)
print("train scaler:", iris_svm_clf_scl_train_acc)
print("test scaler:", iris_svm_clf_scl_test_acc)


# In[21]:


iris_acc = [iris_svm_clf_train_acc, iris_svm_clf_test_acc, iris_svm_clf_scl_train_acc, iris_svm_clf_scl_test_acc]

file_mse_name = "iris_acc.pkl"

open_file = open(file_mse_name, "wb")
pickle.dump(iris_acc, open_file)
open_file.close()

open_file = open(file_mse_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)

