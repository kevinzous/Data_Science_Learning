#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Importing the libraries and files 
# =============================================================================
import pandas as pd
desired_width = 320    
pd.set_option('display.width', desired_width)
pd.set_option('max_colwidth', 100)
pd.options.display.float_format = '{:,.4f}'.format ##or .map({:,.0f})
import numpy as np
import matplotlib.pyplot as plt
#only in notebooks  % matplotlib inline
from matplotlib import style 
style.use('ggplot') 
import seaborn as sns            ##for pairplots..

import statsmodels.formula.api as smf
import statsmodels.api as sm     #for qqplots 
import scipy.stats as stats      #for normal distrib

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prepro
pd.options.display.max_columns=60  

# read the data
train = pd.read_csv('data/P1train.csv', index_col=0)
test = pd.read_csv('data/P1test.csv', index_col=0)
county_facts_dictionary = pd.read_csv('data/county_facts_dictionary.csv')

# =============================================================================
# Preprocessing
# =============================================================================
#Step 1-3 : only the first 52 columns 
train['n']=train['Bernie Sanders']+train['Hillary Clinton']
train.drop(columns=['Bernie Sanders', 'Donald Trump', 'Hillary Clinton','John Kasich', 'Ted Cruz'],inplace=True)
train.HilaryPercent=train.HilaryPercent*100
test.drop(columns=['Bernie Sanders', 'Donald Trump', 'Hillary Clinton','John Kasich', 'Ted Cruz'],inplace=True)

#dropping rows with empty HilaryPercent
train.dropna(inplace=True)
test.dropna(inplace=True)


# load the functions to be used
from util_formula import *

y = 'HilaryPercent'

Cols=[
"Persons 65 years and over, percent, 2014"
,"Female persons, percent, 2014" 
,"Black or African American alone, percent, 2014"
,"High school graduate or higher, percent of persons age 25+, 2009-2013"
,"Persons below poverty level, percent, 2009-2013"]

Features=[ 'Q("'+ x +'")' for x in Cols]
Features[2]='np.sqrt('+Features[2]+')' #np.sqrt(Q("Black or African American alone, percent, 2014"))

seperator = '+'
Features_concat=seperator.join(Features)

#First step #Candidates=['AGE775214','SEX255214','RHI225214','EDU635213','PVY020213']

Candidates =['PST045214', 'PST120214', 'AGE135214', 'AGE295214', 'AGE775214', 'SEX255214', 'RHI125214', 'RHI225214', 'RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214', 'RHI825214', 'POP715213', 'POP645213', 'POP815213', 'EDU635213', 'EDU685213', 'VET605213', 'LFE305213', 'HSG010214',
       'HSG445213', 'HSG096213', 'HSG495213', 'HSD410213', 'HSD310213', 'INC910213', 'INC110213', 'PVY020213', 'BZA010213', 'BZA110213', 'BZA115213', 'NES010213', 'SBO001207', 'SBO315207', 'SBO115207', 'SBO215207', 'SBO515207', 'SBO415207', 'SBO015207', 'MAN450207', 'WTN220207', 'RTN130207', 'RTN131207', 'AFN120207',
       'BPS030214', 'LND110210', 'POP060210']
## removing n and hilary percent, 'Population, 2010 (April 1) estimates base','Population, 2010',

#ols modal
fullmodel = modelFitting(y, Candidates, train)

# enumerate all models and obtain the results
models = pd.DataFrame({"model":[], "SSE": [], "R2":[], "AR2": [], "AIC": [], "BIC": [], "Pnum":[]})
for i in range(3,5):
    models = models.append(getAll(i, y, Candidates, train));
    
# get the Mallow's Cp Statistic
models = getMallowCp(models, fullmodel)

findBest(models, 'R2')

# =============================================================================
# 
# =============================================================================
# use forward selection to get the best model
fwmodel = forward(y, Candidates, train, 'AIC')
fwmodel.summary()
Features_fw=['RHI225214', 'EDU635213', 'RHI625214', 'INC910213', 'HSG495213', 'SBO415207', 'EDU685213', 'AGE775214', 'PST120214', 'RHI825214', 'LND110210', 'SEX255214', 'SBO315207', 'RHI425214', 'POP645213', 'AGE135214', 'HSD310213', 'POP715213', 'RHI325214', 'LFE305213', 'RHI725214', 'HSG445213', 'HSG096213', 'POP060210']
#returns
#AIC:                         1.760e+04
#R-squared:                       0.662
#Adj. R-squared:                  0.659

bwmodel = backward(y, Candidates, train, 'AIC')
bwmodel.summary()
#returns
Features_bw=['PST045214', 'AGE295214', 'RHI325214', 'RHI525214', 'POP815213', 'HSG010214', 'INC110213', 'PVY020213', 'BZA110213', 'BZA115213', 'NES010213', 'SBO115207', 'SBO215207', 'SBO015207', 'MAN450207', 'WTN220207', 'RTN131207', 'AFN120207', 'BPS030214']
#AIC:                         1.757e+04
#R-squared:                       0.668
#Adj. R-squared:                  0.664

## include interactions in the predictors
cand_2Inter = []
for p1 in Candidates:
    for p2 in Candidates:
        if p1 == p2:
            cand_2Inter.append(p1);
        else:
            cand_2Inter.append(p1+':'+p2);
                
print(cand_2Inter)
print(len(cand_2Inter))

# forward selection with interaction considered
fwmodel = forward(y, cand_2Inter, train, 'AIC')
fwmodel.summary()
#R-squared:                       0.768
#Adj. R-squared:                  0.758
#AIC:                         1.685e+04

Features_fw_int=['RHI125214:RHI225214', 'RHI625214:RHI825214', 'RHI825214:HSG096213', 'AGE775214:HSG096213', 'EDU635213', 'SEX255214:INC910213', 'POP715213:HSG495213', 'RHI225214:HSD410213', 'HSG495213:INC110213', 'RHI825214:POP645213', 'RHI725214:LFE305213', 'AGE295214:AGE775214', 'RHI225214:HSD310213', 'SEX255214:RHI225214', 'PST120214:PVY020213', 'AGE135214:INC110213', 'AGE775214:EDU685213', 'RHI125214:RHI425214', 'AGE295214:HSG495213', 'PST120214:RHI825214', 'POP715213:INC910213', 'EDU635213:HSG495213', 'RHI525214:POP060210', 'AGE295214:INC910213', 'SEX255214:HSG495213', 'RHI525214:SBO315207', 'SBO115207:LND110210', 'RHI325214:POP645213', 'AGE135214:HSG445213', 'RHI325214:HSG495213', 'RHI125214:INC910213', 'RHI725214:POP815213', 'AGE295214:POP815213', 'AGE135214:RHI425214', 'AGE775214:BZA115213', 'AGE135214:POP815213', 'INC910213:PVY020213', 'POP815213:HSD310213', 'RHI425214:POP645213', 'PST120214:RHI625214', 'RHI325214:VET605213', 'HSG445213:PVY020213', 'INC910213:INC110213', 'HSG495213:PVY020213', 'AGE295214:RHI225214', 'RHI225214:HSG445213', 'RHI225214:EDU685213', 'PVY020213:SBO515207', 'POP815213:LND110210', 'SBO315207:SBO015207', 'AGE135214:AGE775214', 'RHI725214:HSG495213', 'POP815213:INC110213', 'RHI225214:POP815213', 'HSG096213:SBO315207', 'RHI225214:RHI825214', 'HSG445213:BZA110213', 'RHI225214:NES010213', 'POP645213:SBO315207', 'AGE775214:RHI625214', 'AGE775214:VET605213', 'BZA115213:LND110210', 'HSG445213:WTN220207', 'AGE295214:HSG445213', 'HSG096213:LND110210', 'AGE135214:LND110210', 'LFE305213:LND110210', 'PST120214:HSG495213', 'PST120214:AGE295214', 'PST120214:EDU635213', 'PST120214:HSG445213', 'AGE775214:POP645213', 'EDU685213:INC110213', 'AGE775214:INC110213', 'RHI425214:SBO415207', 'EDU635213:INC910213', 'SEX255214', 'SEX255214:EDU635213', 'MAN450207:RTN130207', 'RHI625214:SBO315207', 'RHI825214', 'RHI225214:SBO001207', 'RHI525214:VET605213', 'RHI525214:RHI725214', 'RHI825214:SBO415207', 'HSG495213:SBO415207', 'RHI525214:POP815213', 'PST120214:AGE775214', 'PST120214:HSD310213', 'AGE775214:RHI225214', 'HSG010214:HSG445213', 'RHI525214:HSG010214', 'AGE135214:EDU635213', 'PST045214:RHI225214', 'RHI325214:HSD310213', 'SEX255214:RHI725214', 'PST120214:INC910213', 'AGE775214:HSG495213', 'AGE775214:PVY020213', 'RHI325214:PVY020213', 'SEX255214:EDU685213', 'EDU685213:PVY020213', 'POP645213:SBO115207', 'BZA115213', 'LFE305213:HSD310213', 'RHI225214:RHI625214', 'AGE775214:EDU635213', 'PST120214:INC110213', 'AGE135214:RHI225214', 'POP815213:HSG495213', 'RHI825214:NES010213', 'RHI425214:BZA115213', 'INC910213:BZA115213', 'EDU635213:EDU685213', 'AGE295214:BZA010213', 'POP815213:RTN130207', 'AGE295214:RHI625214', 'AGE775214:HSG445213', 'PST120214:AGE135214', 'RHI725214:SBO315207', 'EDU685213:AFN120207', 'POP815213:SBO315207', 'HSG445213:HSG495213', 'RHI625214:PVY020213', 'HSD410213:SBO015207', 'EDU685213:HSD310213', 'RHI325214:SBO215207', 'POP645213:SBO215207', 'SBO015207', 'POP715213:HSG010214', 'EDU685213:SBO015207', 'SBO115207:SBO415207', 'HSG096213:SBO015207', 'SBO315207:AFN120207', 'RHI225214:RHI725214', 'POP645213:HSG495213', 'RHI125214:HSG495213', 'AGE135214:NES010213', 'SBO315207:SBO215207', 'PST120214:POP060210', 'SBO315207:SBO415207', 'POP645213:INC910213', 'AGE135214:SBO315207', 'POP645213:RTN131207', 'RHI625214:LFE305213', 'POP715213:RTN131207']

CrossValidation(y,Xfeatures,train,KFold(n_splits=2).split(train))