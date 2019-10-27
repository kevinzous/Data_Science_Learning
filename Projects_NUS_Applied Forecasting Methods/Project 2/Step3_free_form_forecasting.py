##Data Quality 

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


### ARIMA ####

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

