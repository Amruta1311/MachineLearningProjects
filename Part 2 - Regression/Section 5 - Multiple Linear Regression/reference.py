# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:45:27 2019

@author: amrut
"""


#IMPORTING THE LIBRARIES
import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

dataset= pd.read_csv('50_Startups.csv')  #imported the dataset
X = dataset.iloc[:, :-1].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,4].values  #contains thedependent matrix unlike X having the independent matrix

#CATEGORICAL VARIABLES

# we only need numbers so categorical variables need to e encoded ie purchase and country here
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder() # for country categorical variable
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#AVOIDING THE DUMMY VARIABLE TRAP

X=X[:,1:]


#SPLITTING THE DATA TO TRIANING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling since the scales of the age and salary are not equivalent and salary dominates over age
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#FITTING MULTIPLE LINEAR REGRESSION TO THE TRAI ING SET
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#PREDICTING THE TEST SET RESULTS

y_pred=regressor.predict(X_test)

#BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION
import statsmodels.formula.api as sm
#in backward elimination as y=b0+b1*x1+b2*x2+... here b0 is not taken into account by sm so we need to add a column x0 that has all values 1 so that b0*x0 will be a counstant that can be taken into account othrwise b0 remains 0. this is required by the stats model library later for statistical information
X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1)
#when axis is 1 means we are adding a column otherwise if axis =0 then we are adding a row
#X_opt will contain only those set of independent variables that have a high impact on the profit
#step-2
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()

#STEP-3
regressor_OLS.summary()

#STEP-4 AND 5
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0, 3, 5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()



X_opt = X[:, [0, 3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#thus the r&d is an important independent variable for determining the profit 






