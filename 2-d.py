#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from statistics import mean, median
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

BI_data = pd.read_csv('BI_data.csv')

def bayesianInference(data):
    B = mean(data[0:28])
    lbar = 1 / B
    total1 = sum(data[28:35])
    total2 = sum(data[28:42])
    total3 = sum(data[28:49])
    total4 = sum(data[28:56])
    post1 = (7+lbar, total1)
    post2 = (14+lbar, total2)
    post3 = (21+lbar, total3)
    post4 = (28+lbar, total4)
    return (post1, post2, post3, post4)

dat = list(BI_data['Daily Deaths'][30:])
end = max(dat[0:28])
tup = bayesianInference(list(BI_data['Daily Deaths']))
l = []
for i, pair in enumerate(tup):
    beta = pair[0] + 1
    alpha = pair[1]
    x = np.linspace(0, int(end), 10000)
    y = stats.gamma.pdf(x, alpha, scale=1/beta)
    map1 = (alpha - 1) / beta
    print(str(i+5) + "th Week MAP: " + str(map1))
    l.append((x,y,map1))
    
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