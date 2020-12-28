# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:16:11 2019

@author: amrut
"""

#SIMPLE LINEAR REGRESSION
#The goal is to form a correlation between the years of experience and the salary so that we can compare the true salary with the predicted salary. Thus from the correlation the mdoel will be able to give some prediction for each
#of the no of years of experience and we will see how the predictions will be close to the true results 


#IMPORTING THE LIBRARIES
import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

dataset= pd.read_csv('Salary_Data.csv')  #imported the dataset
X = dataset.iloc[:, :-1].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,1].values  #contains thedependent matrix unlike X having the independent matrix

#SPLITTING THE DATA TO TRIANING AND TEST SETS
# here we do 20 observations in the training set and 10 observations in the test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=1/3, random_state=0)


#Feature Scaling is not applied in the linear regression as the library is ready to take care of that
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''


#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
#By doing the following our model will be able to learn the correlation of our training set to learn how it can predict the dependent variable based on the independent variable

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
#y_pred is the vector of predictions in our machinery model
#y_test contains the real salarys whereas y_pred contains the predicted salary
y_pred=regressor.predict(X_test)

#VISUALISING THE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('YEARS OF EXPERIENCE')
plt.ylabel('SALARY')
plt.show()

#VISUALISING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
# the above line of code is not changed since even if we replace it as plt.plot(X_test, regressor.predict(X_test), color='blue') will give us the same line of regression
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('YEARS OF EXPERIENCE')
plt.ylabel('SALARY')
plt.show()



