#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import pickle


# In[2]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)


df.plot(x='x',y='y', height=600)


# In[3]:


data = pd.read_csv('./dane_do_regresji.csv')
# data


# In[4]:


train, test = train_test_split(data, test_size=.2)


# In[5]:


# print(train)


# In[6]:


# print(test.x)


# In[7]:


#POLY TRAIN LIN
lin_reg =  LinearRegression()
lin_reg.fit(train[['x']], train['y'])
lin_reg_train = lin_reg.predict(train[['x']])
# print(lin_reg.intercept_, lin_reg.coef_, "\n", lin_reg.predict(train[['x']]))
lin_reg_train_mse = mean_squared_error(train[['y']], lin_reg_train)
print(lin_reg_train_mse)


# In[8]:


#POLY TEST LIN
lin_reg_test = lin_reg.predict(test[['x']])
# print(lin_reg.intercept_, lin_reg.coef_, "\n", lin_reg.predict(test[['x']]))
lin_reg_test_mse = mean_squared_error(test[['y']], lin_reg_test)
print(lin_reg_test_mse)


# In[9]:


#POLY TRAIN KNN=3
knn_reg_3 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_reg_3.fit(train[['x']], train['y'])
knn_3_reg_train = knn_reg_3.predict(train[['x']])
# print(knn_reg.predict(train[['x']]))
knn_3_reg_train_mse = mean_squared_error(train[['y']], knn_3_reg_train)
print(knn_3_reg_train_mse)


# In[10]:


#POLY TEST KNN=3
knn_3_reg_test = knn_reg_3.predict(test[['x']])
# print(knn_reg.predict(test[['x']]))
knn_3_reg_test_mse = mean_squared_error(test[['y']], knn_3_reg_test)
print(knn_3_reg_test_mse)


# In[11]:


#POLY TRAIN KNN=5
knn_reg_5 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_reg_5.fit(train[['x']], train['y'])
knn_5_reg_train = knn_reg_5.predict(train[['x']])
# print(knn_reg.predict(train[['x']]))
knn_5_reg_train_mse = mean_squared_error(train[['y']], knn_5_reg_train)
print(knn_5_reg_train_mse)


# In[12]:


#POLY TEST KNN=5
knn_5_reg_test = knn_reg_5.predict(test[['x']])
# print(knn_reg.predict(test[['x']]))
knn_5_reg_test_mse = mean_squared_error(test[['y']], knn_5_reg_test)
print(knn_5_reg_test_mse)


# In[13]:


#POLY TRAIN DEGREE=2
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features_2.fit_transform(train[['x']])
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly, train[['y']])
# print(lin_reg.intercept_, lin_reg.coef_)
poly_2_train = poly_2_reg.predict(poly_features_2.fit_transform(train[['x']]))
# print(lin_reg.predict(poly_features.fit_transform(train[['x']])))
poly_2_train_mse = mean_squared_error(train[['y']], poly_2_train)
print(poly_2_train_mse)


# In[14]:


#POLY TEST DEGREE=2
X_poly = poly_features_2.fit_transform(test[['x']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_2_test = poly_2_reg.predict(poly_features_2.fit_transform(test[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(test[['x']])))
poly_2_test_mse = mean_squared_error(test[['y']], poly_2_test)
print(poly_2_test_mse)


# In[15]:


#POLY TRAIN DEGREE=3
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features_3.fit_transform(train[['x']])
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly, train[['y']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_3_train = poly_3_reg.predict(poly_features_3.fit_transform(train[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(train[['x']])))
poly_3_train_mse = mean_squared_error(train[['y']], poly_3_train)
print(poly_3_train_mse)


# In[16]:


#POLY TEST DEGREE=3
X_poly = poly_features_3.fit_transform(test[['x']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_3_test = poly_3_reg.predict(poly_features_3.fit_transform(test[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(test[['x']])))
poly_3_test_mse = mean_squared_error(test[['y']], poly_3_test)
print(poly_3_test_mse)


# In[17]:


#POLY TRAIN DEGREE=4
poly_features_4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features_4.fit_transform(train[['x']])
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly, train[['y']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_4_train = poly_4_reg.predict(poly_features_4.fit_transform(train[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(train[['x']])))
poly_4_train_mse = mean_squared_error(train[['y']], poly_4_train)
print(poly_4_train_mse)


# In[18]:


#POLY TEST DEGREE=4
X_poly = poly_features_4.fit_transform(test[['x']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_4_test = poly_4_reg.predict(poly_features_4.fit_transform(test[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(test[['x']])))
poly_4_test_mse = mean_squared_error(test[['y']], poly_4_test)
print(poly_4_test_mse)


# In[19]:


#POLY TRAIN DEGREE=5
poly_features_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features_5.fit_transform(train[['x']])
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly, train[['y']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_5_train = poly_5_reg.predict(poly_features_5.fit_transform(train[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(train[['x']])))
poly_5_train_mse = mean_squared_error(train[['y']], poly_5_train)
print(poly_5_train_mse)


# In[20]:


#POLY TEST DEGREE=5
X_poly = poly_features_5.fit_transform(test[['x']])
# print("INTERCEPT, COEF:", lin_reg.intercept_, lin_reg.coef_)
poly_5_test = poly_5_reg.predict(poly_features_5.fit_transform(test[['x']]))
# print("PREDICT, FIT TRASNFORM: \n", lin_reg.predict(poly_features.fit_transform(test[['x']])))
poly_5_test_mse = mean_squared_error(test[['y']], poly_5_test)
print(poly_5_test_mse)


# In[21]:


mse_data = {'train_mse':[lin_reg_train_mse, knn_3_reg_train_mse, knn_5_reg_train_mse, poly_2_train_mse, poly_3_train_mse, poly_4_train_mse, poly_5_train_mse], 'test_mse':[lin_reg_test_mse, knn_3_reg_test_mse, knn_5_reg_test_mse, poly_2_test_mse, poly_3_test_mse, poly_4_test_mse, poly_5_test_mse]} 
mse_df_index = pd.DataFrame(mse_data, index =['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])  


# In[22]:


file_mse_name = "mse.pkl"

open_file = open(file_mse_name, "wb")
pickle.dump(mse_df_index, open_file)
open_file.close()

open_file = open(file_mse_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[23]:


reg_data = [(lin_reg, None), (knn_reg_3, None), (knn_reg_5, None), (poly_2_reg, poly_features_2), (poly_3_reg, poly_features_3), (poly_4_reg, poly_features_4), (poly_5_reg, poly_features_5)]


# In[24]:


file_mse_name = "reg.pkl"

open_file = open(file_mse_name, "wb")
pickle.dump(reg_data, open_file)
open_file.close()

open_file = open(file_mse_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)


# In[25]:


print(poly_2_reg.coef_[0][1] * 2**2 + poly_2_reg.coef_[0][0] * 2 + poly_2_reg.intercept_[0])
#y = -5.47661005 * x2 + 3.37146732 * x + 6.62738022

