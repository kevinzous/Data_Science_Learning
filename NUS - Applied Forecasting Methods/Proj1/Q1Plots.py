# Run import
#%run "1-Train.py"
from Q1Train import *
# =============================================================================
# Diagnosis of linear regression
# =============================================================================

# check the normality of the residuals
fig = plt.figure(figsize=(12,12))
fig = sm.qqplot(lm.resid, stats.distributions.norm, line='r') 

# residuals against fitter value
plt.figure(figsize=(10,8))
plt.scatter(lm.fittedvalues,lm.resid)


# partial plots
fig1 = plt.figure(figsize=(40,20))
fig1 = sm.graphics.plot_regress_exog(lm, Features[0], fig=fig1) ## big variance but normal

fig2 = plt.figure(figsize=(40,20))
fig2 = sm.graphics.plot_regress_exog(lm, Features[1], fig=fig2) ## big variance and when predic increases

fig3 = plt.figure(figsize=(40,20))
fig3 = sm.graphics.plot_regress_exog(lm, Features[2], fig=fig3)
#putting sqrt deleted the curvature but still have variance issue
fig1 = plt.figure(figsize=(40,20))
fig1 = sm.graphics.plot_regress_exog(lm, Features[3], fig=fig1)
# big variance but normal
fig1 = plt.figure(figsize=(40,20))
fig1 = sm.graphics.plot_regress_exog(lm, Features[4], fig=fig1) 
# big variance but normal

# pair plot of some variables of interest
Dic_inv_sub={k:Dic_inv[k] for k in Cols if k in Dic_inv}
sns.set(font_scale=1)

b=sns.pairplot(data=train[Cols].rename(columns=Dic_inv_sub),
               diag_kind='kde',
               height=2, aspect=2, #Size of 1 plot, width = height* aspect
               kind='reg',)

fig.suptitle('Pair plots of variables of interest', 
              fontsize=12, 
              fontweight='bold')


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
