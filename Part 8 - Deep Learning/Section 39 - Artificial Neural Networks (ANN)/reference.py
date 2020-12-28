# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:09:07 2019

@author: amrut
"""

#PArt 1 Data Preprocessing


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 Lets make the ANN

#Import the Keras Libraries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN

classifier = Sequential()

#Adding the input layer and the first hidden layer -->output_dim=(11+1)/2 = 6 where 11 are the input layers 1 is the output layer and the 6 is the hidden layer
#activation fucntion chosen is rectifier fucntion
# Moreover init initialises the weights and sees that the weights remain small numbers close to zero

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

#Adding the second hidden layer 
# for the second hidden layer we need not expect any input parameters since it knows what to expect because of the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#COmpiling the ANN that is using the Stochastic Gradient Decent

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to the Training set

classifier.fit(X_train, y_train, batch_size =10, nb_epoch =100 )


#Part 3 Making the predictions and evaluating the model


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy is (1891+261)/2500 = 86.08% that the test set provides the right results
