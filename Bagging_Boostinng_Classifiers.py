# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:43:18 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Bagging and Boosting\\iris.csv")

X=df.iloc[:,0:4]
y=df.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#modelbuilding with Decision tree

model=DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)


#Modelbuilding with BoostingClassifier

model2=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0, n_estimators=20)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred2)

#modelbuilding with RandomForest

model3=RandomForestClassifier(n_estimators=20)
model3.fit(X_train,y_train)
y_pred=model3.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)


#modelbuilding with Adaboost classifier

model4=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)



















