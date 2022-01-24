import numpy as np
import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

a=load_wine()
X = a.data
y = a.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(metrics.confusion_matrix(y_test,y_pred))
