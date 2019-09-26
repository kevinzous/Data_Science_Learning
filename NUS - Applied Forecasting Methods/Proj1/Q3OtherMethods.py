#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##OTHER METHODS

##INCLUDE STATES

#linear regression 
#polynomial regression

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))


cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())