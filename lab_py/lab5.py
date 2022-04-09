#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys

from subprocess import call

import graphviz
from sklearn.tree import export_graphviz

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import pickle


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print('LOADED DATA FROM "data_breast_cancer"')


# In[3]:


cancer_data = pd.concat([data_breast_cancer["target"], data_breast_cancer["data"]['mean texture'], data_breast_cancer["data"]['mean symmetry']], axis=1)


# In[4]:


print('|     N     | MAX_DEPTH |      MSE_TRAIN     |     MSE_TEST    |')
print('----------------------------------------------------------------')
max_depths_amounts = np.zeros((20), dtype=int)
for j in range(20):
    train, test = train_test_split(cancer_data, test_size=.2)
    max_train_score = 0
    max_test_score = 0
    max_depth = 0
    for i in range(20):
        if(i == 0):
            tree_clf_perm = DecisionTreeClassifier()
            tree_clf_perm.fit(train[['mean texture',  'mean symmetry']], train['target'])
        else:
            tree_clf_perm = DecisionTreeClassifier(max_depth=i)
            tree_clf_perm.fit(train[['mean texture',  'mean symmetry']], train['target'])

        y_train_pred_perm = tree_clf_perm.predict(train[['mean texture',  'mean symmetry']])
        y_test_pred_perm = tree_clf_perm.predict(test[['mean texture',  'mean symmetry']])

        cancer_score_train_perm = f1_score(train['target'], y_train_pred_perm)
        cancer_score_test_perm = f1_score(test['target'], y_test_pred_perm)
        if((cancer_score_train_perm < 1 and cancer_score_test_perm < 1) and (max_train_score < cancer_score_train_perm and max_test_score < cancer_score_test_perm)):
            max_train_score = cancer_score_train_perm
            max_test_score = cancer_score_test_perm
            max_depth = i
    print(f'{j+1}-th optimal max_depth={max_depth+1} {max_train_score}, {max_test_score}')
    max_depths_amounts[max_depth] =  max_depths_amounts[max_depth] + 1
print('-------------------------------------------------------------')    
print(max_depths_amounts)
print('-------------------------------------------------------------')

max_depth_max_count = 0
max_depth_max_count_index = None

for index,count  in enumerate(max_depths_amounts):
    if(max_depth_max_count < count):
        max_depth_max_count_index = index
        max_depth_max_count = count
        
print(f'{max_depth_max_count}/20 times the optimal was max_depth={max_depth_max_count_index+1}')


# In[5]:


train, test = train_test_split(cancer_data, test_size=.2)
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(train[['mean texture',  'mean symmetry']], train['target'])
y_train_pred = tree_clf.predict(train[['mean texture',  'mean symmetry']])
y_test_pred = tree_clf.predict(test[['mean texture',  'mean symmetry']])

# print(data_breast_cancer.target_names,"\n",  data_breast_cancer['target'])
# print(y_train_pred)


# In[6]:


print('\n \n---------------TRAIN DATA---------------')
print(train)
print('\n-------------------------------TRAIN PREDICT------------------------------')
print(y_train_pred)


# In[7]:


print('\n \n---------------TEST DATA----------------')
print(test)
print('\n-------------------------------TEST PREDICT-------------------------------')
print(y_test_pred)


# In[8]:


f = 'bc.dot'
export_graphviz(
    tree_clf,
    out_file=f,
    feature_names=cancer_data.columns[:2],
    class_names=[str(num)+", "+name 
                     for num, name in zip(set(cancer_data['target']), cancer_data.columns[1])],
    rounded=True,
    filled=True
)
call(['dot', '-Tpng', 'bc.dot', '-o', 'bc.png'])
graph = graphviz.Source.from_file(f)
graph


# In[9]:


cancer_score_train_maxdepth3 = f1_score(train['target'], y_train_pred)
cancer_score_test_maxdepth3 = f1_score(test['target'], y_test_pred)
print('--------------F1 SCORE--------------')
print(cancer_score_train_maxdepth3, cancer_score_test_maxdepth3)


# In[10]:


cancer_acc_train_maxdepth3 = accuracy_score(train['target'], y_train_pred)
cancer_acc_test_maxdepth3 = accuracy_score(test['target'], y_test_pred)
print('--------------ACCURANCY--------------')
print(cancer_acc_train_maxdepth3, cancer_acc_test_maxdepth3)


# In[11]:


f1acc_tree = [3, cancer_score_train_maxdepth3, cancer_score_test_maxdepth3, cancer_acc_train_maxdepth3, cancer_acc_test_maxdepth3]

file_f1acc_tree_name = "f1acc_tree.pkl"

open_file = open(file_f1acc_tree_name, "wb")
pickle.dump(f1acc_tree, open_file)
open_file.close()

open_file = open(file_f1acc_tree_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('--------------------------------------EXERCICE 4----------------------------------')
print(loaded_list)


# In[21]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')
print('----------------LOADED DATA FROM LOCAL VARS----------------')


# In[13]:


print('|     N     | MAX_DEPTH |     MSE_TRAIN     |     MSE_TEST   |')
print('--------------------------------------------------------------')
max_depths_amounts = np.zeros((40), dtype=int)
for j in range(40):
    min_train_mse = sys.float_info.max
    min_test_mse = sys.float_info.max
    max_depth = 0
    max_sub = 0
    for i in range(40):
        train_reg_perm, test_reg_perm = train_test_split(df, test_size=.2)
        if(i == 0):
            tree_reg_perm = DecisionTreeRegressor()
            tree_reg_perm.fit(train_reg_perm[['x']], train_reg_perm['y'])
        else:
            tree_reg_perm = DecisionTreeRegressor(max_depth=i)
            tree_reg_perm.fit(train_reg_perm[['x']], train_reg_perm['y'])

        y_train_reg_pred_perm = tree_reg_perm.predict(train_reg_perm[['x']])
        y_test_reg_pred_perm = tree_reg_perm.predict(test_reg_perm[['x']])

        mse_tree_reg_train_perm = mean_squared_error(train_reg_perm['y'], y_train_reg_pred_perm)
        mse_tree_reg_test_perm = mean_squared_error(test_reg_perm['y'], y_test_reg_pred_perm)
        
        if((mse_tree_reg_train_perm < 100 and mse_tree_reg_test_perm < 100) and (mse_tree_reg_train_perm > 0.0 and mse_tree_reg_test_perm > 0.0)):
            if((mse_tree_reg_train_perm < min_train_mse and mse_tree_reg_test_perm < min_test_mse)):
                min_train_mse = mse_tree_reg_train_perm
                min_test_mse = mse_tree_reg_test_perm
                max_depth = i
            elif(((mse_tree_reg_train_perm <= min_train_mse and mse_tree_reg_test_perm + 5 <= min_test_mse) or (mse_tree_reg_train_perm + 5 <= min_train_mse and mse_tree_reg_test_perm <= min_test_mse))):
                min_train_mse = mse_tree_reg_train_perm
                min_test_mse = mse_tree_reg_test_perm
                max_depth = i
                
    print(f'{j+1}-th  optimal max_depth={max_depth+1} {min_train_mse}, {min_test_mse}')
    max_depths_amounts[max_depth] =  max_depths_amounts[max_depth] + 1
print('-------------------------------------------------------------')    
print(max_depths_amounts)
print('-------------------------------------------------------------')
max_depth_max_count = 0
max_depth_max_count_index = None

for index,count  in enumerate(max_depths_amounts):
    if(max_depth_max_count < count):
        max_depth_max_count_index = index
        max_depth_max_count = count       
print(f'{max_depth_max_count}/40 times the optimal was max_depth={max_depth_max_count_index+1}')


# In[14]:


train_reg_3, test_reg_3 = train_test_split(df, test_size=.2)
tree_reg_3 = DecisionTreeRegressor(max_depth=3)
tree_reg_3.fit(train_reg_3[['x']], train_reg_3['y'])
y_train_reg_3_pred = tree_reg_perm.predict(train_reg_3[['x']])
y_test_reg_3_pred = tree_reg_perm.predict(test_reg_3[['x']])


# In[15]:


print('\n \n--------TRAIN DATA------')
print(train_reg_3)
print('\n--------------------------TRAIN PREDICT-------------------------')
print(y_train_reg_3_pred)


# In[16]:


print('\n \n--------TRAIN DATA------')
print(test_reg_3)
print('\n--------------------- -----TRAIN PREDICT-------------------------')
print(y_test_reg_3_pred)


# In[17]:


f = 'reg.dot'
export_graphviz(
    tree_reg_3,
    out_file=f,
    feature_names=df.columns[0],
    class_names=[str(num)+", "+name 
                     for num, name in zip(set(df['y']), df.columns[1])],
    rounded=True,
    filled=True
)
call(['dot', '-Tpng', 'reg.dot', '-o', 'reg.png'])
graph_reg = graphviz.Source.from_file(f)
graph_reg


# In[18]:


mse_tree_reg_3_train = mean_squared_error(train_reg_3['y'], y_train_reg_3_pred)
mse_tree_reg_3_test = mean_squared_error(test_reg_3['y'], y_test_reg_3_pred)
print('-----------------MSE--------------')
print(mse_tree_reg_3_train, mse_tree_reg_3_test)


# In[19]:


tree_reg_mse = [4, mse_tree_reg_3_train, mse_tree_reg_3_test]

file_mse_name = "mse_tree.pkl"

open_file = open(file_mse_name, "wb")
pickle.dump(tree_reg_mse, open_file)
open_file.close()

open_file = open(file_mse_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print('------------------EXERCICE 4--------------')
print(loaded_list)


# In[ ]:




