# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:56:10 2019

@author: amrut
"""

#This is a random selection Algorithm which we have implemented before implementing UCB since we want to understand the advantage of using UCB
#Upper COnfidence Bound --> CLoser to AI as it has a dynamic strategy and it depends on the observations from the beginning of the experiment up to the present time

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()