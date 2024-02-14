import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import re
import nltk
from rake_nltk import Rake, Metric
from scipy.stats import uniform, randint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# packages for performance metrics
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import data
df = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal\\Data\\stability-data-all.csv")

# drop unnecessary columns
X = df
X = X.drop(columns=['sentiment_negative', 'sentiment_neutral ', 'sentiment_positive', 'sentiment_NA',
                    'not_relevant', 'relevant', 'very_relevant', 'extremely_relevant', 'Sentences'])
X = X.drop(columns=['Sentences'])

# Split to  train and test data
train, test = train_test_split(X, test_size=0.3, random_state=0, shuffle=True)

X_train = train
X_train = X_train.drop(columns=['target'])
y_train = train.loc[:, 'target'].values

X_test = test
X_test = X_test.drop(columns=['target'])
y_test = test.loc[:, 'target'].values

# LOGISTIC REGRESSION
lg = LogisticRegression(solver='lbfgs')
lg.fit(X_train, y_train)
pred = lg.predict(X_test)

print("Basic Logistic Regression")
print(classification_report(y_test,pred))
print("ROC-AUC score: ", roc_auc_score(y_test, pred))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
pred = svm.predict(X_test)

print("Basic SVM")
print(classification_report(y_test, pred))
print("ROC-AUC score: ", roc_auc_score(y_test, pred))

# RANDOM FOREST
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Basic RF")
print(classification_report(y_test, pred))
print("ROC-AUC score: ", roc_auc_score(y_test, pred))

# NN
nn = MLPClassifier()
nn.fit(X_train, y_train)
pred = nn.predict(X_test)

print('Basic NN')
print(classification_report(y_test, pred))
print("ROC-AUC score: ", roc_auc_score(y_test, pred))