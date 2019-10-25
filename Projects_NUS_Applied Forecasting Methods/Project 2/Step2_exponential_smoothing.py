# =============================================================================
# # import the packages and libraries
# =============================================================================
import pandas as pd
pd.options.display.max_columns=60

import numpy as np
import sys

import statsmodels.tsa.statespace as sts

from matplotlib import style
style.use('ggplot')

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.arima_process as sta
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

from util_formula import *

# =============================================================================
# 1. reading data 
# =============================================================================
train = pd.read_csv('data/P2train.csv', parse_dates=['Time'],index_col='Time',header=0)
test = pd.read_csv('data/P2test.csv', parse_dates=['Time'],index_col='Time',header=0)
test_index = pd.read_csv('data/P2test_index.csv', header=0)

print(train.shape)
train.head(10)


# =============================================================================
# #moving average
# =============================================================================
df=train['2017-02-01']
y='TrafficVolume'

fig, ax = plt.subplots(1,3,figsize=(20,5))
win = 2;
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')

win = 3; 
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')

win = 4;
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[2].scatter(x=range(0,df.shape[0]), y=df[y])
ax[2].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')



df=train
y='TrafficVolume'

fig, ax = plt.subplots(1,3,figsize=(20,5))
win = 24;
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')

win = 168; # 24*7
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')

win = 720; # 24*30
ma = df[y].rolling(window=win, min_periods=None, center=False)
ax[2].scatter(x=range(0,df.shape[0]), y=df[y])
ax[2].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='red')



# =============================================================================
# #exponentially weighted moving average
# =============================================================================
fig, ax = plt.subplots(1,3,figsize=(15,5))
al = 0.1;
ewma = df[y].ewm(alpha=al, min_periods=0)
ax[0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='red')

al = 0.3;
ewma = df[y].ewm(alpha=al, min_periods=0)
ax[1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='red')

al = 0.5;
ewma = df[y].ewm(alpha=al, min_periods=0)
ax[2].scatter(x=range(0,df.shape[0]), y=df[y])
ax[2].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='red')

# =============================================================================
# #selecting the smoothing parameters
# =============================================================================
alpha = np.linspace(0.01,1,num=100)
err = [];
for al in alpha:
    ewma = df[y].ewm(alpha=al, min_periods=0)
    pred = ewma.mean();
    diff = df[y] - pred.shift(1);
    err.append(np.sqrt((diff ** 2).mean()))
    
plt.plot(alpha, err)
optal = alpha[np.argmin(err)]
plt.axvline(x=optal, color='red')
print(optal)

# =============================================================================
# #2. Trend Corrected Smoothing
# =============================================================================

# given a series and alpha, return series of smoothed points
def double_exponential_smoothing(series, alpha, beta, L0, B0):
    result = []
    for n in range(0, len(series)):
        val = series[n]
        if n==0:
            level = alpha*val + (1-alpha)*(L0+B0);
            trend = beta*(level-L0) + (1-beta)*B0;
            last_level = level;
        else:
            level = alpha*val + (1-alpha)*(last_level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            last_level = level;
            
        result.append(level)
    return result

a = 0.4;
b = 0.1;
series = df[y].values
holt = double_exponential_smoothing(series, a, b, series[0], series[1]-series[0])
fig = plt.figure(figsize=(8,8))
plt.scatter(x=range(0,df.shape[0]), y=df[y])
plt.scatter(x=range(0,df.shape[0]), y=holt, color='red')

