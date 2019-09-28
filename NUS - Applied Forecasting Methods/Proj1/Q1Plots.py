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
sns.set(font_scale=1)
        
diagnosisplot(modal,Features)
#plots for simple modal with sqrt 
diagnosisplot(modalsqrt,Featuressqrt)

# =============================================================================
# ## Outliers and influence points 
# =============================================================================

fig, ax = plt.subplots(figsize=(14,10))
fig = sm.graphics.influence_plot(modal, ax=ax, criterion="cooks")

fig1, ax = plt.subplots(figsize=(20,10))
fig1=sm.graphics.plot_leverage_resid2(modal,ax=ax)

#low leverage influential points 
    #positive res
        #NEBRASKA
        #1883 NE 
        #1724 NE
        #1008 NE 
        #39 NE 
        #COLORADO
        #912 CO
    #negative res
        #1126 ID
        #1369 UT
        #2341 UT
#high leverage and high influential points 
#469 GA  GA=GEORGIA
#113 TX


## check the leverage and influential points
influence = modal.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
fig,ax = plt.subplots(figsize=(10,6))
#ax.set
fig = plt.stem(np.arange(len(c)), c, markerfmt=",",use_line_collection=True) 
plt.ylabel("Cook's distance")
plt.xlabel('index')

### Checking some potential outilers points 
util_formula.meaning(train).iloc[c.argmax()]

util_formula.meaning(train).loc['469 GA']
util_formula.meaning(train).loc['1883 NE']
util_formula.meaning(train).loc['1008 NE']
util_formula.meaning(train).loc['2341 UT']
util_formula.meaning(train).loc['1369 ID']

# =============================================================================
# # Collinearity
# =============================================================================
# pair plot of some variables of interest
sns.set(font_scale=1)
b=sns.pairplot(data=train[Features],diag_kind='kde',height=2, 
aspect=2, #Size of 1 plot, width = height* aspect
kind='reg')
fig.suptitle('Pair plots of variables of interest',fontsize=12, fontweight='bold')

#Variance inflation factor
for i in range(6):
     print(sms.outliers_influence.variance_inflation_factor(modal.model.exog, i))

inf = sms.outliers_influence.OLSInfluence(modal)
inf.summary_table()

train_=util_formula.Getwsme('HilaryPercent',train,Formula,'1st step')[1]

###Log linear transformation
plt.hist(np.log(train['HilaryPercent']+12000))
plt.scatter(x=train['SEX255214'],y=np.log(train['HilaryPercent']+2000))

# =============================================================================
# Bias in prediction 
# ==============================================================
#ploting the wsme against predictors

len(list(trainfitsqrt.columns))-3

for i in range(2):
    xs=trainfitsqrt[list(trainfitsqrt.columns)[i]]
    ys=trainfitsqrt['wmse']
    labels=list(trainfitsqrt.index)

    for i,label in enumerate(labels):
        x = xs[i]
        y = ys[i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+0.3, y+0.3, str(i)+' '+label, fontsize=9)
        plt.legend("dsd)
    plt.show()


plt.scatter(x, y, marker='x', color='red')
plt.text(x+0.3, y+0.3, str(i)+' '+label, fontsize=9)

plt.show()

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
