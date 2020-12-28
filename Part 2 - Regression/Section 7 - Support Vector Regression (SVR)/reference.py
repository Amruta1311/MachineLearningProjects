# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:08:45 2019

@author: amrut
"""

#SVR - Support Vector Regression
# They support linear and non linear regressions
#SVR  performs linear regression at higher dimensional space
#We can think of SVR as if each data point in the training represents its own dimension.
#When you evaluate your kernel between the test point and a point in the training set the resulting value gives you the coordinate of your test point in that dimension
#The vector we get when we evaluate the test bpoint for all points in the training set, be it k bar , is the representation of the test point in the higher dimensional space
#Once we get the k bar vector we can perform the linear regression
#In the classification problem, the vextors x bar are used to define a hyperplane that separates the two different classes in your solution
#These vectors are used to perform linear regression. The vectors closest to the test point are referred to as support vectors.
#We can perform our fucntion anywhere so any vectors could be closest to our test evaluation location
# Building a SVR
# 1. Collect a training set T={X bar, Y bar}
# 2. Coose a kernel and its parameters as well as any regularization needed
# 3. Form the correlation matrix, K bar
# 4. Train your machine, exactly or approximately , to get contraction coefficients alpha bar = {alpha i}
# 5. Use those coefficients, create your estimator f(X bar, alpha bar , x*)=y*
# We need to choose a prominent kernel. One of the kernels are the Gaussian Kernel, Noise Kernel. Along with the kernel the regularization is also important since it helps prevent wild fluctuations between data points
# Next series of pics on ipad
# All together our goal is to make the errors not exceed the threshold in SVR unlike the linear regression where we try to minimize the error between the prediction and data

# Regression Template

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




































 