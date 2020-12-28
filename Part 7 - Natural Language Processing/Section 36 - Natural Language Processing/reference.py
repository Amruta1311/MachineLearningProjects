# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:13:44 2019

@author: amrut
"""
#Natural Language Processing
#Used for Analysing texts that can be book reviews or HTML webpages etc

#Importing Datasets 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting= 3) #By using quoting =3 we are ignoring the double quotes

#Cleaning the Texts--> That is working with texts and making the model to predict things on text

import re #has some great tools to clean efficently some texts
import nltk #Helps in removing the words that are prepositions or ones that dont help much in the review and are wastes
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#corpus is the collection of text of the same type
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])       # Will help in only keeping the letters in the review and remove all the numbers and punctuations
    review = review.lower()
    review = review.split()
    #Doing the stemming process that helps in getting the root of the words like the root of the word loved or lovely is 'LOVE'
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the Bag of worlds Model

#BAg Of Words Model --> we take all the different words present in our corpus and create one column for each word. Since there are a lot of words we will have a lot of columns 
#Thuse we have a table of 1000 rows and a lot of columns for each words
#Matrix containing a lot of zeroes is called the sparse matrix
#Creating a sparse matrix is the bag of words model itself
#Tokenization is the process of taking all the different words in the review and creating each column for each word 
#Why do we need such a mdoel? Because simply what we do in the end is to predict if a review is positive or not and for a machine learning model to e able to predict that will need to be trained for all these reviews
#Here we are doing nothing but classification where we have some independent variables on which we train our machine on the model to predict the dependent variables which is the categorical variable that is the binary outcome

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  #max_features helps is selecting the main words that willhelp us judge a review rather than waste words that do not add to and efiicency
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

'''#Using the Classification Model Naive Bayes 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy of NAive BAyes Model is (55+91)/200  = 73%  '''


'''#Using the Classification Model Decision Tree

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy of Decision Tree Model is (74+68)/200  = 71%  '''


#Using the Classification Model Random Forest 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy of Decision Tree Model is (87+57)/200  = 72%