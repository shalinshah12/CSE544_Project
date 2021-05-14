#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from statistics import mean, median
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

BI_data = pd.read_csv('data/Bayesian_Inference_data.csv')

def bayesianInference(data):
    """Computes the parameters alpha and beta that 
        describe our posterior distribution
        
        Args:
        data - a list of combined daily deaths of the
               months of February and March
        """
    
    B = mean(data[0:28]) # Mean of first four weeks of February
    lbar = 1 / B # Denotes the Initial lambda parameter of the Exponential Prior distribution (Given as inverse of sample mean of februrary)
    alpha1, beta1 = sum(data[28:35]), 7+lbar
    alpha2, beta2 = sum(data[28:42]), 14+lbar
    alpha3, beta3 = sum(data[28:49]), 21+lbar
    alpha4, beta4 = sum(data[28:56]), 28+lbar
    post1 = (beta1, alpha1) # Parameters used to define the posterior distribution; we name them Beta and Alpha
    post2 = (beta2, alpha2)
    post3 = (beta3, alpha3)
    post4 = (beta4, alpha4)
    return (post1, post2, post3, post4)

dat = list(BI_data['Daily Deaths'][30:])
end = max(dat[0:28])
tup = bayesianInference(list(BI_data['Daily Deaths']))
l = []
for i, pair in enumerate(tup):
    #Get params for every posterior distribution
    beta = pair[0] + 1
    alpha = pair[1]
    x = np.linspace(0, int(end), 10000)
    y = stats.gamma.pdf(x, alpha, scale=1/beta) #PDF of every posterior distribution
    map1 = (alpha - 1) / beta #MAP for the distribution
    print(str(i+5) + "th Week MAP: " + str(map1))
    l.append((x,y,map1))
    
#Plot all posterior distributions on the same graph
fig, ax = plt.subplots()
ax.plot(l[0][0], l[0][1], label="5th Week Posterior" )
ax.plot(l[1][0], l[1][1], label="6th Week Posterior")
ax.plot(l[2][0], l[2][1], label="7th Week Posterior")
ax.plot(l[3][0], l[3][1], label="8th Week Posterior")
ax.legend(loc = 'upper right')
plt.title('Posterior Distributions')
plt.xlabel('X')
plt.ylabel('PDF')
plt.show()