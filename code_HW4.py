# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from sklearn.model_selection import train_test_split

## descriptive statistics (exploratory data analysis)

data = pd.read_csv("C:/Users/user/Box/IE 517 Machine Learning in FIN Lab/HW4/housing.csv", header='infer')

##  CRIM: Per capita crime rate by town
# ZN : Proportion of residential land zoned for lots over 25,000 sq. ft.
# INDUS : Proportion of non-retail business acres per town
# CHAS : Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX : Nitric oxide concentration (parts per 10 million)
# RM : Average number of rooms per dwelling
# AGE : Proportion of owner-occupied units built prior to 1940
# DIS : Weighted distances to five Boston employment centers
# RAD : Index of accessibility to radial highways
# TAX : Full-value property tax rate per $10,000
# PTRATIO : Pupil-teacher ratio by town
# B : 1000(Bk - 0.63)^2, where is the proportion of [people of African American descent] by town
# LSTAT : Percentage of lower status of the population
# MEDV: Median value of owner-occupied homes in $1000s

## overall view of this data
data.head

## there are 506 rows and 14 columns

########################################### Exploratory Data Analysis ########################################################
# print summary of data frame
summary = data.describe()
print(summary)

## scatterplot matrix
variables = data[["CRIM", "INDUS", "NOX", "RM", "TAX", "PTRATIO", "LSTAT", "MEDV"]]
sns.pairplot(variables, size = 3)
plt.tight_layout()
plt.show()

## correlation matrix

corr_matrix = np.corrcoef(variables.values.T)
hm = sns.heatmap(corr_matrix,
                 cbar=True,
                 annot=True,
                 square=True,
                 yticklabels=["CRIM", "INDUS", "NOX", "RM", "TAX", "PTRATIO", "LSTAT", "MEDV"],
                 xticklabels=["CRIM", "INDUS", "NOX", "RM", "TAX", "PTRATIO", "LSTAT", "MEDV"])
plt.show()

## house price distribution
    
_ = plt.hist(data['MEDV'])
_ = plt.xlabel('MEDV')
_ = plt.ylabel('count')
plt.show()

## house price distribution based on the index of accessability of radial highway (RAD)

MEDV = data['MEDV']
MEDV1 = MEDV[data['RAD']==1]
MEDV2 = MEDV[data['RAD']==2]
MEDV3 = MEDV[data['RAD']==3]
MEDV4 = MEDV[data['RAD']==4]
MEDV5 = MEDV[data['RAD']==5]
MEDV6 = MEDV[data['RAD']==6]
MEDV7 = MEDV[data['RAD']==7]
MEDV8 = MEDV[data['RAD']==8]
MEDV24 = MEDV[data['RAD']==24]

MEDV_array = [MEDV1, MEDV2, MEDV3, MEDV4, MEDV5, MEDV6, MEDV7, MEDV8, MEDV24]

_ = plt.boxplot(MEDV_array)
_ = plt.xlabel("RAD")
_ = plt.ylabel("Hosing Price")
plt. show()

## bee swarm plot

_ = sns.swarmplot(x='RAD', y='MEDV', data=data)

_ = plt.xlabel('RAD')
_ = plt.ylabel('Housing Price')

plt.show()


############################################# Spliting training and test sets ############################################

## all features

X, y = data.iloc[:,1:13].values, data["MEDV"]
# Split the dataset into a training and a testing set

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
print( X_train.shape, y_train.shape, X_test.shape, y_test.shape)

############################################ Linear Regression #######################################################

############################################# fit a regression model ################################################

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

## slope
lr.coef_

## intercept
lr.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

## Calculate the MSE

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

##################################################### Ridge regression ################################################
## alpha =1
from sklearn.linear_model import Ridge
ridge_r = Ridge(alpha=1.0)
ridge_r.fit(X,y)
y_train_pred = ridge_r.predict(X_train)
y_test_pred = ridge_r.predict(X_test)

## slope

ridge_r.coef_

## intercept

ridge_r.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

## alpha = 0.5

from sklearn.linear_model import Ridge
ridge_r = Ridge(alpha=0.5)
ridge_r.fit(X,y)
y_train_pred = ridge_r.predict(X_train)
y_test_pred = ridge_r.predict(X_test)

## slope

ridge_r.coef_

## intercept

ridge_r.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

## alpha = 10

from sklearn.linear_model import Ridge
ridge_r = Ridge(alpha=10)
ridge_r.fit(X,y)
y_train_pred = ridge_r.predict(X_train)
y_test_pred = ridge_r.predict(X_test)

## slope

ridge_r.coef_

## intercept

ridge_r.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

## let's do a loop

alpha=[0.001,0.01,0.1,1,10,100,1000]

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


me=[]
r2=[]
for i in np.arange(len(alpha)):
    ridge_r = Ridge(alpha=alpha[i])
    ridge_r.fit(X,y)
    y_train_pred = np.array(ridge_r.predict(X_train))
    y_test_pred = np.array(ridge_r.predict(X_test))
    me.append(mean_squared_error(y_test, y_test_pred))
    r2.append(r2_score(y_test, y_test_pred))

plt.plot(alpha, me, linestyle='--')
plt.xlabel('alpha value')
plt.ylabel('MSE')

plt.plot(alpha, r2, linestyle='--')
plt.xlabel('alpha value')
plt.ylabel('R-square')




##################################################### LASSO ################################################
## alpha =1
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X,y)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

## slope

lasso.coef_

## intercept

lasso.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

## alpha = 0.5

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.5)
lasso.fit(X,y)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

## slope

lasso.coef_

## intercept

lasso.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))


## alpha = 10

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=10)
lasso.fit(X,y)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

## slope

ridge_r.coef_

## intercept

ridge_r.intercept_

## ploting the residual figures
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white',
            label='Test data')

from sklearn.metrics import mean_squared_error
print('MSE train: %3f, test: %.3f' %(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

## calculate R-squre
from sklearn.metrics import r2_score

print('R^2 train: %.3f, test: %.3f' %
      (r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))


## let's do a loop

alpha=[0.001,0.01,0.1,1,10,100,1000]

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


me=[]
r2=[]
for i in np.arange(len(alpha)):
    lasso = Lasso(alpha=1.0)
    lasso.fit(X,y)
    y_train_pred = np.array(lasso.predict(X_train))
    y_test_pred = np.array(lasso.predict(X_test))
    me.append(mean_squared_error(y_test, y_test_pred))
    r2.append(r2_score(y_test, y_test_pred))

plt.plot(alpha, me, linestyle='--')
plt.xlabel('alpha value')
plt.ylabel('MSE')

plt.plot(alpha, r2, linestyle='--')
plt.xlabel('alpha value')
plt.ylabel('R-square')




## ending
print("My name is Richie Ma")
print("My NetID is: ruchuan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")