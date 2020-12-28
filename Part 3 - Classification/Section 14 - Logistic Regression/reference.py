# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:52:18 2019

@author: amrut
"""

#LOGISTIC REGRESSION

#IMPORTING THE LIBRARIES
import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

# before importing the dataset we need to specify the working directory folder
#We can set any folder as the working folder as long as they have the data.csv file

dataset= pd.read_csv('Social_Network_Ads.csv')  #imported the dataset
X = dataset.iloc[:,2:4].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,4].values  #contains thedependent matrix unlike X having the independent matrix


#SPLITTING THE DATA TO TRIANING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=0)
#by test_size the split is done as 2 observations of X are in the test set and 8 will be in the training set

#Feature Scaling since the scales of the age and salary are not equivalent and salary dominates over age
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#FITTING THE LOGISTIC REGRESSION INTO OUR TRAINING SET

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
#thus our classifer will learn the relation between the training set of X and y and will use this to predict the observations


#Predicting the Test set results

y_pred = classifier.predict(X_test)


#MAKING THE CONFUSION MATRIX
#THIS HELPS IN EVALUATING THE PREDICTIVE POWER OF OUR LOGISTIC REGRESSION MODEL
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
# in the cm matrix we have 69+84 correct predictions and we have 8+3 ie 11 incorrect predictions and thuse that percent will be reflected on the graph

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

























