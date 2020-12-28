# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:47:18 2019

@author: amrut
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mall.csv')

X = dataset.iloc[:,[3,4]].values

#Using the elbbow method to find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans =KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=10, random_state = 0)
    #n_cluster is taken as i since we will be trying the elbow method for different values of i
    #init is taken as k-means++ since we need to avoid the random initialisation trap 
    #n_init is taken as 10 to that the kmeans alogrithm will run 10 times with different initial centroids
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No Of Clusters')
plt.ylabel('WCSS')
plt.show()

#Thus our K = 5

# Applying the kmeans to the mall dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state= 0 )
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s =100, c='red',label='Careful')

plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s =100, c='blue',label='Standard')

plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s =100, c='green',label='Targets')

plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s =100, c='cyan',label='Careless')

plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s =100, c='magenta',label='Sensible')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s =300, c='yellow',label='Centroids')

plt.title('Clusters Of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()