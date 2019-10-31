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
from statsmodels.tsa.stattools import acf, pacf,kpss,adfuller

from util_formula import *


# =============================================================================
# Data Quality 
# =============================================================================

train.loc['2017-02-28 23'] ##gives 2 values 

#date columns remove the hour of the date time index
train['date'] = [train.index[i].date() for i in range(0,len(train))]

#count number of values by day 
train[['date','TrafficVolume']].groupby(by='date').count().sort_values(by='TrafficVolume') #values ranges from 81 to 1
train.loc['2012-12-16'] ## 81 
train.loc['2013-08-31']
#### for Temparature, we have one temp per hour, for weather each time it changes, we have a new values 


##same for test set : 
test['date'] = [test.index[i].date() for i in range(0,len(test))]

#count number of values by day 
test[['date','TrafficVolume']].groupby(by='date').count().sort_values(by='TrafficVolume') #values ranges from 81 to 1

dates = pd.date_range(start='2012-01-01',end='2017-01-01',freq='H')
df = pd.DataFrame(np.random.randn(len(dates), 1), index=dates, columns=['A'])
monthly_mean = df.resample('M').mean()

# =============================================================================
# stationarity test KPSS
# =============================================================================
'''Kwiatkowski-Phillips-Schmidt-Shin test for stationarity:
Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null hypothesis that x is level or trend stationary.'''

df=df['2017']
result = kpss(df.TrafficVolume.values, regression='c') # c H0=data is stationary around a constant (default).
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#The p-value is greater than 0.05. The null hypothesis of stationarity around a cste is not rejected! 


result = kpss(df.TrafficVolume.values, regression='ct') #ct:  H0=data is stationary around a trend
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#The p-value is smaller than 0.05. The null hypothesis of stationarity around a trend is rejected. #not stationary around trend since has no trend 
    
    
# =============================================================================
# Augmented Dickey Fuller test (ADH Test)
# =============================================================================
'''Augmented Dickey-Fuller unit root test

The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.
The null hypothesis of the Augmented Dickey-Fuller is that there is a 
unit root.'''

result = adfuller(df.TrafficVolume.values, autolag='AIC',regression='c')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
## null hypothesis rejected, meaning that the datasets is stationary ! 


# =============================================================================
# differencing if not stationary
# =============================================================================

# plot the moving average/std with window size = period
df=train
y='TrafficVolume'
df=df[[y]]

df=df['2017-02']
fig, ax = plt.subplots(1,1,figsize=(22,6))
ax.scatter(x=range(0,df.shape[0]), y=df[y])
ax.axis('tight')

# take the differencing to make it more stationary
dif1 = df - df.shift(1)
dif2 = df - df.shift(2)
fig, ax = plt.subplots(1,2,figsize=(22,6))
ax[0].scatter(x=range(0,dif1.shape[0]), y=dif1[y])
ax[0].axis('tight')
ax[1].scatter(x=range(0,dif2.shape[0]), y=dif2[y])
ax[1].axis('tight')
plt.show()

# =============================================================================
#  Estimating ARMA parameters 
# =============================================================================
# =============================================================================
# ACF AND PACF
# =============================================================================
alpha=0.05 ## 95 % for the confidence window
lags=150
y='TrafficVolume'
df=train[[y]]  ##double bracket to have a dataframe type instead of series type

data = df['2017']
fig, ax = plt.subplots(2,1,figsize=(22,12))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')
# =============================================================================
# Nonseasonal modeling
# =============================================================================

