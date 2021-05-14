#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from statistics import mean, median
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv('data/clean_organised.csv')
feb_data = pd.read_csv('data/feb_data.csv')
march_data = pd.read_csv('data/march_data.csv')

def std(data):
    """Computes corrected standard deviation
        of a dataset in the form of a list"""
    n = len(data)
    u = mean(data)
    total = 0
    for x in data:
        total += (x-u) * (x-u)    
    total /= (n-1)
    total = total ** 0.5
    return total

def oneSampleWaldTest(data1, data2):
    """Computes w statistic for One sample
        Wald Test and makes a decision on
        the basis of a threshold"""
    threshold = 1.96
    theta0 = mean(data1)
    n = len(data2)
    s = sum(data2)
    u = mean(data2)
    se_hat = (u/n) ** 0.5
    w = (u - (theta0)) / (se_hat)
    if abs(w) > threshold:
        return "Reject"
    return "Accept"

def twoSampleWaldTest(data1, data2):
    """Computes w statistic for Two sample
        Wald Test and makes a deicison on
        the basis of a threshold"""
    threshold = 1.96
    n, m = len(data1), len(data2)
    diff = mean(data1) - mean(data2)
    m1, m2 = mean(data1), mean(data2)
    se_hat =  ((m1/n) + (m2/m)) ** 0.5
    w = diff / se_hat
    if abs(w) > threshold:
        return "Reject"
    return "Accept"

def oneSampleZTest(data1, data2, sigma):
    """Computes Z statistic for one sample
        Wald Test and makes a deicison on
        the basis of a threshold"""
    #sigma = true stdev of data2
    threshold = 1.96
    u0 = mean(data1)
    xbar = mean(data2)
    n = len(data2)
    rootn = n ** 0.5
    z = (xbar - u0) / (sigma/rootn)
    if abs(z) > threshold:
        return "Reject"
    return "Accept"

def oneSampleTTest(data1, data2):
    """Computes T statistic for One sample
        T Test and makes a deicison on
        the basis of a threshold"""
    threshold = 2.042
    u0 = mean(data1)
    xbar = mean(data2)
    n = len(data2)
    rootn = n ** 0.5
    sigma = std(data2)
    t = (xbar - u0) / (sigma/rootn)
    if abs(t) > threshold:
        return "Reject"
    return "Accept"

def twoSampleTTest(data1, data2):
    """Computes T statistic for Two sample
        T Test and makes a deicison on
        the basis of a threshold"""
    threshold = 2 # approx value of n = 60 instead of n = 57
    n, m = len(data1), len(data2)
    diff = mean(data1) - mean(data2)
    sigmax = std(data1)
    sigmay = std(data2)
    t = diff / (((sigmax/n)+(sigmay/m))**0.5)
    if abs(t) > threshold:
        return "Reject"
    return "Accept"

sigma1 = std(df['MT daily cases'])
sigma2 = std(df['NC daily cases'])
sigma3 = std(df['MT daily death'])
sigma4 = std(df['NC daily death'])

d1, d2 = list(feb_data['MT daily cases']), list(march_data['MT daily cases'])
print("One Sample Wald Test for Montana Cases: " + oneSampleWaldTest(d1, d2))
print("Two Sample Wald Test for Montana Cases: " + twoSampleWaldTest(d1, d2))
print("One Sample Z Test for Montana Cases: " + oneSampleZTest(d1, d2, sigma1))
print("One Sample T Test for Montana Cases: " + oneSampleTTest(d1, d2))
print("Two Sample T Test for Montana Cases: " + twoSampleTTest(d1, d2))

d1, d2 = list(feb_data['NC daily cases']), list(march_data['NC daily cases'])
print("One Sample Wald Test for NC Cases: " + oneSampleWaldTest(d1, d2))
print("Two Sample Wald Test for NC Cases: " + twoSampleWaldTest(d1, d2))
print("One Sample Z Test for NC Cases: " + oneSampleZTest(d1, d2, sigma2))
print("One Sample T Test for NC Cases: " + oneSampleTTest(d1, d2))
print("Two Sample T Test for NC Cases: " + twoSampleTTest(d1, d2))

d1, d2 = list(feb_data['MT daily death']), list(march_data['MT daily death'])
print("One Sample Wald Test for Montana Deaths: " + oneSampleWaldTest(d1, d2))
print("Two Sample Wald Test for Montana Deaths: " + twoSampleWaldTest(d1, d2))
print("One Sample Z Test for Montana Deaths: " + oneSampleZTest(d1, d2, sigma3))
print("One Sample T Test for Montana Deaths: " + oneSampleTTest(d1, d2))
print("Two Sample T Test for Montana Deaths " + twoSampleTTest(d1, d2))

d1, d2 = list(feb_data['NC daily death']), list(march_data['NC daily death'])
print("One Sample Wald Test for NC Deaths: " + oneSampleWaldTest(d1, d2))
print("Two Sample Wald Test for NC Deaths: " + twoSampleWaldTest(d1, d2))
print("One Sample Z Test for NC Deaths: " + oneSampleZTest(d1, d2, sigma4))
print("One Sample T Test for NC Deaths: " + oneSampleTTest(d1, d2))
print("Two Sample T Test for NC Deaths: " + twoSampleTTest(d1, d2))