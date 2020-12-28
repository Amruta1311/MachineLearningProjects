# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:57:12 2019

@author: amrut
"""

# RANDOM FOREST INTUITIONS --> Based on decision Tree
# 1. Pick at random K data points from the training set
# 2. Build the decision tree associated to these K data points
# 3. Choose the number Ntrees of trees you want to build and repeat steps 1 and 2
# 4. For a new data point, make each one of your Ntrees predict the value of Y to for the data point in question and assign the new data point the average across all of the predicted Y values
 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Random Forest Regression  Model to the dataset
# Create your regressor here

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
#On increasing or decreasing the nestimator we will not get more or less no of steps but rather they wiill be more better placed coorresponding to the dependent variable axis
regressor.fit(X,y)


# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Compared to the Decision Tree Regression, the Random FOrest Regression has more no of stairs and thuse there are more splits and a lot more intervals in this



