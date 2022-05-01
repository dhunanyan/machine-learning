#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[3]:


cancer_data = pd.concat([data_breast_cancer["target"], data_breast_cancer["data"]['mean texture'], data_breast_cancer["data"]['mean symmetry']], axis=1)
train, test = train_test_split(cancer_data, test_size=.2)

X_train = train[['mean texture',  'mean symmetry']]
y_train = train['target']
X_test = test[['mean texture',  'mean symmetry']]
y_test = test['target']


# In[4]:


print(X_train)
print(y_train)


# In[5]:


print(X_test)
print(y_test)


# In[6]:


log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
k_neigh_clf = KNeighborsClassifier()

voting_clf_h = VotingClassifier(
    estimators=[('lr', log_clf),
                ('tc', tree_clf), 
                ('knc', k_neigh_clf)], 
    voting='hard')

voting_clf_s = VotingClassifier(
    estimators=[('lr', log_clf),
                ('tc', tree_clf), 
                ('knc', k_neigh_clf)], 
    voting='soft')


# In[7]:


acc_list = []
for i, clf in enumerate((log_clf, tree_clf, k_neigh_clf, voting_clf_h, voting_clf_s)):
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    acc_list.append((
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred))
    )
    print(f"{clf.__class__.__name__}: ({accuracy_score(y_train, y_train_pred)}, {accuracy_score(y_test, y_test_pred)})")


# In[8]:


file_acc_name = "acc_vote.pkl"

open_file = open(file_acc_name, "wb")
pickle.dump(acc_list, open_file)
open_file.close()

open_file = open(file_acc_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 4-----------------------------------')
print(loaded_list)


# In[9]:


clf_list = [log_clf, tree_clf, k_neigh_clf, voting_clf_h, voting_clf_s]
file_clf_name = "vote.pkl"

open_file = open(file_clf_name, "wb")
pickle.dump(clf_list, open_file)
open_file.close()

open_file = open(file_clf_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 4-----------------------------------')
print(loaded_list)


# In[10]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 30,
    bootstrap=True
)
bag_half_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 30,
    max_samples = 0.5,
    bootstrap=True
)
past_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 30,
    bootstrap=False
)
past_half_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 30,
    max_samples = 0.5,
    bootstrap=False
)
rnd_clf = RandomForestClassifier(n_estimators=30)
ada_boost_clf = AdaBoostClassifier(n_estimators=30)
gbrt_clf = GradientBoostingClassifier(n_estimators=30)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[11]:


acc_bag_list = []
for i, clf in enumerate((bag_clf, bag_half_clf, past_clf, past_half_clf, rnd_clf, ada_boost_clf, gbrt_clf)):
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    acc_bag_list.append((
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred))
    )
    print(f"{clf.__class__.__name__}: ({accuracy_score(y_train, y_train_pred)}, {accuracy_score(y_test, y_test_pred)})")


# In[12]:


file_acc_bag_name = "acc_bag.pkl"

open_file = open(file_acc_bag_name, "wb")
pickle.dump(acc_bag_list, open_file)
open_file.close()

open_file = open(file_acc_bag_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 6-----------------------------------')
print(loaded_list)


# In[13]:


clf_bag_list = [bag_clf, bag_half_clf, past_clf, past_half_clf, rnd_clf, ada_boost_clf, gbrt_clf]
file_clf_name = "bag.pkl"

open_file = open(file_clf_name, "wb")
pickle.dump(clf_bag_list, open_file)
open_file.close()

open_file = open(file_clf_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 4-----------------------------------')
print(loaded_list)


# In[14]:


print(train)


# In[15]:


df1 = pd.DataFrame(data_breast_cancer.data, columns=data_breast_cancer.feature_names)
df1['target'] = data_breast_cancer.target
X2 = df1.iloc[:, 0:30]
y = data_breast_cancer.frame.target
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2)


# In[16]:


acc_fea_list = []

fea_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators = 30,
    max_samples = 0.5,
    bootstrap=True,
    bootstrap_features=True, 
    max_features=2
)
fea_clf.fit(X2_train, y2_train)

y2_train_pred_fea = fea_clf.predict(X2_train)
y2_test_pred_fea = fea_clf.predict(X2_test)
acc_fea_list.append((
    accuracy_score(y2_train, y2_train_pred_fea),
    accuracy_score(y2_test, y2_test_pred_fea))
)


# In[17]:


file_acc_fea_name = "acc_fea.pkl"

open_file = open(file_acc_fea_name, "wb")
pickle.dump(acc_fea_list, open_file)
open_file.close()

open_file = open(file_acc_fea_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 8-----------------------------------')
print(loaded_list)


# In[18]:


fea_clf_list = [fea_clf]
file_clf_name = "fea.pkl"

open_file = open(file_clf_name, "wb")
pickle.dump(fea_clf_list, open_file)
open_file.close()

open_file = open(file_clf_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 8-----------------------------------')
print(loaded_list)


# In[19]:




# y2_train_pred_0 = fea_clf.estimators_[0].predict(X2_train)
# y2_test_pred_0 = fea_clf.estimators_[0].predict(X2_test)


# In[20]:


data_fea_list = []
for est_fea, est in zip(fea_clf.estimators_features_, fea_clf.estimators_):
    y_train_pred_temp = est.predict(X2_train.iloc[:, est_fea])
    y_test_pred_temp = est.predict(X2_test.iloc[:, est_fea])
    data_fea_list.append([accuracy_score(y2_train, y_train_pred_temp), accuracy_score(y2_test, y_test_pred_temp), X2_train.iloc[:, est_fea].columns.tolist()])
    print(f"{X2_train.iloc[:, est_fea].columns.tolist()}: ({accuracy_score(y2_train, y_train_pred_temp)}, {accuracy_score(y2_test, y_test_pred_temp)})")
                      
df_fea = pd.DataFrame(data_fea_list, columns = ['train_accuracy', 'test_accuracy', 'features_names'])             


# In[21]:


df_fea_sorted = df_fea.sort_values(by=['train_accuracy', 'test_accuracy'], ascending = False)
df_fea_sorted


# In[22]:


file_acc_fea_rank_name = "acc_fea_rank.pkl"

open_file = open(file_acc_fea_rank_name, "wb")
pickle.dump(df_fea_sorted, open_file)
open_file.close()

open_file = open(file_acc_fea_rank_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('---------------------------------------EXERCICE 9-----------------------------------')
print(loaded_list)


# In[ ]:




