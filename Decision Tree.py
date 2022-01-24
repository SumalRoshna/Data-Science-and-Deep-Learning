import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('iris.csv')
print(df.columns)
x= df.iloc[:, [2,3]].values  
y= df.iloc[:, 4].values 
print(y)


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 


from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)  
y_pred=classifier.predict(x_test)
print(y_pred)


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
print(cm)


from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(y_test,y_pred))
from sklearn import tree
tree.plot_tree(classifier)
