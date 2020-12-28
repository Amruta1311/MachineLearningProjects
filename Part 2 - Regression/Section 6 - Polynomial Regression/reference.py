# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 08:52:45 2019

@author: amrut
"""

#POLYNOMIAL REGRESSION
# LINEAR REGRESSION --> b0 + b1*x1
# MULTIPLE REGRESSION --> b0 + b1*x1 + b2*x2 +b3*x3 + ... + bn*xn
# POLYNOMIAL LINEAR REGRESSION  --> b0 + b1*x1 + b2*(x1^2) + b2*(x1^3) + ... + bn*(x1 ^n)
# when the dataset does not fit that well in a simple regression model becasue it resembles a polynomial like structure on the grapg then we prefer polynomial regression since it is more fitting
#polynomial regression is used in cases where we need to find out hoe epidemic sreads accross a territory

#Here we will be building a bluffing detector using polynomial regression

#IMPORTING THE LIBRARIES
import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

dataset= pd.read_csv('Position_Salaries.csv')  #imported the dataset
X = dataset.iloc[:, 1:2].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,2].values  #contains thedependent matrix unlike X having the independent matrix
#it is always better to keep X as a matrix and y as a vector 


#SPLITTING THE DATA TO TRIANING AND TEST SETS
#since we are having only 10 observations thus we need to convert it into a training and test set since we do not have enough information for the model on one set and test its performance on the other set
# Moreover we even need to make an accurate prediction and not miss the target thus we need to have maximum information as possible for training our machine elarning mdoel


#Feature Scaling 
#polynomial regression consists of adding some polynomial terms into the multiple regression equation and therefore we use the same linear regression library that we used in simple and multiple linear regression models

#FITTING THE LINEAR REGRESSION TO THE DATASET
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)



#FITTING THE POLYNOMIAL REGRESSION TO THE DATASET
from sklearn.preprocessing import PolynomialFeatures
# poly_reg will add another column of matrix of features X^2,X^3,...,X^n
poly_reg=PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#VISUALISING THE LINEAR REGRESSION RESULTS
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth Or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#VISUALISING THE POLYNOMIAL REGRESSION RESULTS

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
#poly_reg.fit_transform(X) is used instead of X_poly since we can have it on any matrix of feature X
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#PREDICTING A NEW RESULT WITH LINEAR REGRESSION

lin_reg.predict(6.5)


#PREDICTING A NEW RESULT WITH POLYNOMIAL REGRESSION

lin_reg_2.predict(poly_reg.fit_transform(6.5))











