#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from statistics import mean, median
import scipy
import numpy as np

df = pd.read_csv('14.csv')

n = len(df)

def oneSampleWaldTest(data1, data2):
    threshold = 1.96
    theta0 = mean(data1)
    n = len(data2)
    s = sum(data2)
    w = (s - (n*theta0)) / (s**0.5)
    if abs(w) > threshold:
        return false
    return true

def twoSampleWaldTest(data1, data2):
    threshold = 1.96
    n, m = len(data1), len(data2)
    diff = mean(data1) - mean(data2)
    m1, m2 = mean(data1), mean(data2)
    se =  ((m1/n) + (m2/m)) ** 0.5
    w = diff / se
    if abs(w) > threshold:
        return false
    return true

def oneSampleZTest(data1, data2, sigma):
    #sigma = true stdev of data2
    threshold = 1.96
    u0 = mean(data1)
    xbar = mean(data2)
    n = len(data)
    rootn = n ** 0.5
    z = (xbar - u0) / (sigma/rootn)
    if abs(z) > threshold:
        return false
    return true

def twoSampleZTest(data1, data2, sigmax, sigmay):
    # sigmax = true stdev of data1
    # sigmay = true stdev of data2
    threshold = 1.96
    n, m = len(data1), len(data2)
    diff = mean(data1) - mean(data2)
    z = diff / (((sigmax/n)+(sigmay/m))**0.5)
    if abs(z) > threshold:
        return false
    return true

def oneSampleTTest(data1, data2):
    threshold = 2.042
    u0 = mean(data1)
    xbar = mean(data2)
    n = len(data)
    rootn = n ** 0.5
    sigma = std(data2)
    t = (xbar - u0) / (sigma/rootn)
    if t > threshold:
        return false
    return true

def twoSampleTTest(data1, data2):
    threshold = 2 # approx value of n = 60 instead of n = 57
    n, m = len(data1), len(data2)
    diff = mean(data1) - mean(data2)
    sigmax = std(data1)
    sigmay = std(data2)
    t = diff / (((sigmax/n)+(sigmay/m))**0.5)
    if t > threshold:
        return false
    return true

dates = []
for val in df['Date']:
    if(int(val.split('-')[1]) == 2 and int(val.split('-')[0]) == 2021):
        dates.append(val)
feb_data = df[df['Date'].isin(dates)]
feb_data = feb_data.reset_index(drop=True)
dates = []
for val in df['Date']:
    if(int(val.split('-')[1]) == 3 and int(val.split('-')[0]) == 2021):
        dates.append(val)
march_data = df[df['Date'].isin(dates)]
march_data = march_data.reset_index(drop=True)


def std(data):
    n = len(data)
    u = mean(data)
    total = 0
    for x in data:
        total += (x-u)*(x-u)    
    total /= (n-1)
    total = total ** 0.5
    return total

sigma1 = std(df['MT confirmed'])
sigma2 = std(df['NC confirmed'])
sigma3 = std(df['MT deaths'])
sigma4 = std(df['NC deaths'])

dates = []
for val in df['Date']:
    if(int(val.split('-')[1]) in [6,7]):
        dates.append(val)
BI_data = df[df['Date'].isin(dates)]
BI_data = BI_data.reset_index(drop=True)
BI_data['Total'] = BI_data['MT confirmed'] + BI_data['NC confirmed'] + BI_data['MT deaths'] + BI_data['NC deaths']

def bayesianInference(data):
    B = mean(data[0:7])
    lbar = 1 / B
    total1 = sum(data[7:14])
    total2 = sum(data[7:21])
    total3 = sum(data[7:28])
    total4 = sum(data[7:35])
    c = 1 # placeholder
    # After every iteration, the posterior takes the form c * lambda
    post1 = (total1, 7+lbar)
    post2 = (total2, 14+lbar)
    post3 = (total3, 21+lbar)
    post4 = (total4, 28+lbar)
    print(post1, post2, post3, post4)




