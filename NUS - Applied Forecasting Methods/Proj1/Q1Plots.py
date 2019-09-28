# Run import
#%run "1-Train.py"
from Q1Train import *
# =============================================================================
# Diagnosis of linear regression
# =============================================================================

def diagnosisplot(lm,Features):
    '''plotting Histogram of normalized residuals
       quantile-quantile plot of the residuals
       residuals against fitted value
       partial plots'''
    #1-1Histogram of normalized residuals
    res = lm.resid
    f1 = plt.figure(figsize=(8,6))
    f1 = plt.hist(lm.resid_pearson,bins=20)
    f1 = plt.ylabel('Count')
    f1 = plt.xlabel('Normalized residuals') 

    #1-2 check the normality of the residuals
    #quantile-quantile plot of the residuals
    fig2 = plt.figure(figsize=(10,10))
    fig = sm.qqplot(lm.resid, stats.distributions.norm, line='r') 
    
    #1-3 residuals against fitted value
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    ax.scatter(lm.fittedvalues,lm.resid)
    ax.axhline(y=0, linewidth=2, color = 'g')
    ax.set(xlabel='fitted values',ylabel='residuals')
    
    #2 partial plots
    for i in range(0,len(Features)):
        fig1 = plt.figure(figsize=(20,10))
        fig1 = sm.graphics.plot_regress_exog(lm, Features[i],fig=fig1)
        
        
#plots for simple modal 
diagnosisplot(modal,Features)
#plots for simple modal with sqrt 
diagnosisplot(modalsqrt,Featuressqrt)

###Log linear transformation
plt.hist(np.log(train['HilaryPercent']+12000))
plt.scatter(x=train['SEX255214'],y=np.log(train['HilaryPercent']+2000))
train


# pair plot of some variables of interest
sns.set(font_scale=1)
b=sns.pairplot(data=train[Features],diag_kind='kde',height=2, 
aspect=2, #Size of 1 plot, width = height* aspect
kind='reg')
fig.suptitle('Pair plots of variables of interest',fontsize=12, fontweight='bold')


# ============================================================================
# TRial
# ============================================================================
train['Poor'] = train['Persons below poverty level, percent, 2009-2013']>24

b=sns.pairplot(data=train[Cols+['Poor']].rename(columns=Dic_inv_sub),
               diag_kind='kde',
               hue='Poor',
               size=1.8, aspect=1.8, 
               kind='reg',)

fig.suptitle('Pair plots of variables of interest', 
              fontsize=12, 
              fontweight='bold')


fig, ax = plt.subplots(1,1,figsize=(20, 14))
plt.scatter(train[Cols[0]],train['HilaryPercent'])

#plotting only county with higher Black concentration

filter= train['Black or African American alone, percent, 2014']>20
fig, ax = plt.subplots(1,1,figsize=(20, 14))
plt.scatter((train[filter])[Cols[0]],(train[filter])['HilaryPercent'])

#train.plot(kind='scatter', y='HilaryPercent', x=Cols[1], ax=ax)
plt.plot(kind='scatter', y=train['HilaryPercent'], 
         #x=train[Cols[1]], #ax=ax
         )

## Outliers and influence points 
fig, ax = plt.subplots(figsize=(40,20))
fig = sm.graphics.influence_plot(lm, ax=ax, criterion="cooks")

fig1, ax = plt.subplots(figsize=(8,6))
fig1 = sm.graphics.plot_leverage_resid2(lm, ax=ax)


## check the leverage and influential points
influence = lm.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
fig = plt.figure(figsize=(6,6))
fig = plt.stem(np.arange(len(c)), c, markerfmt=",") 
train.iloc[c.argmax()]
