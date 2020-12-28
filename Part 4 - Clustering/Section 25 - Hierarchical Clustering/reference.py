# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:57:10 2019

@author: amrut
"""
import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd

dataset = pd.read_csv('mall.csv')

X = dataset.iloc[:,[3,4]].values

# Using the Dendrogram to find the Optimal No of clusters

import scipy.cluster.hierarchy as sch

dendrogram =sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()




# From the dendrogram we see the otimal no of clusters are 5 which is same as that we found in kmeans 

# Fitting Heirarchical Clustering to the mall dataset

# Linkage minimises the variance in each of the clusters

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s =100, c='red',label='Careful')

plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s =100, c='blue',label='Standard')

plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s =100, c='green',label='Targets')

plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s =100, c='cyan',label='Careless')

plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s =100, c='magenta',label='Sensible')

plt.title('Clusters Of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()



