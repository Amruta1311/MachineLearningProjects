# -*- coding: utf-8 -*-
"""
Created on Mon May 27 01:54:42 2019

@author: amrut
"""

#Library is a tool that is used to make a specific job
#Data Preprocessing

#There are three essential libraries here that need to be imported

import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

# before importing the dataset we need to specify the working directory folder
#We can set any folder as the working folder as long as they have the data.csv file

dataset= pd.read_csv('Data.csv')  #imported the dataset
X = dataset.iloc[:, :-1].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,3].values  #contains thedependent matrix unlike X having the independent matrix

#Take care Missing data
# we replace the empty cells with the mean of the entire column

from sklearn.preprocessing import Imputer #imputer class allows us to take care of the missing data
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) #ctrl + I to see parameters
imputer= imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])

# we only need numbers so categorical variables need to e encoded ie purchase and country here
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder() # for country categorical variable
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#here we need to avoid the situation of france greater than germany and so on thus we use dynamic encoding
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#in dependent variable (purchase) we wont be using onehotencoder since the machine elarning model will know that it is a category 

labelencoder_y=LabelEncoder() # for country categorical variable
y=labelencoder_y.fit_transform(y)


# We make two different sets - one is the training set on which the machine elarning model is built
# and the other is the test set-we test the perofrmance of this machine elarning model
#Preformance between the test and the training set should not be very different 

#SPLITTING THE DATA TO TRIANING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=0)
#by test_size the split is done as 2 observations of X are in the test set and 8 will be in the training set

#Feature Scaling since the scales of the age and salary are not equivalent and salary dominates over age

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




