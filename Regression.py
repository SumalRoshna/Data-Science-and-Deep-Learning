import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('salary.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

regression=LinearRegression()
regression.fit(X_train,y_train)
pred=regression.predict(X_test)

plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,pred)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regression.predict(X_train))
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

r2=r2_score(y_test,pred)
print('r2_score: ',r2)

regression.predict(pd.DataFrame([11],columns=['YearsExperience']))
