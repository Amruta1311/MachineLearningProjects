#IMPORTING THE LIBRARIES
import numpy as np   #contains all the mathematical tools
import matplotlib.pyplot as plt #Helps plot nice charts 
import pandas as pd  # best library to import data sets and manage them


#Importing the DataSets

# before importing the dataset we need to specify the working directory folder
#We can set any folder as the working folder as long as they have the data.csv file

dataset= pd.read_csv('Data.csv')  #imported the dataset
X = dataset.iloc[:, :-1].values  #colon before the comma means include all the lines and colon after comma means to include all columns except the last one. Thus we select our independent variables
y = dataset.iloc[:,3].values  #contains thedependent matrix unlike X having the independent matrix


# We make two different sets - one is the training set on which the machine elarning model is built
# and the other is the test set-we test the perofrmance of this machine elarning model
#Preformance between the test and the training set should not be very different 

#SPLITTING THE DATA TO TRIANING AND TEST SETS
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=0)
#by test_size the split is done as 2 observations of X are in the test set and 8 will be in the training set

#Feature Scaling since the scales of the age and salary are not equivalent and salary dominates over age
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''






