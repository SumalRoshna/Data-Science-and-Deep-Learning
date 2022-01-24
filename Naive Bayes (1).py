#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as nm
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# In[35]:


db=load_breast_cancer()
x=db.data
y=db.target
print(x)
print(y)


# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


# In[37]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
print(x_train)
x_test=sc.transform(x_test)
print(x_test)


# In[38]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
print(classifier)


# In[39]:


y_pred=classifier.predict(x_test)
print(y_pred)


# In[40]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[41]:


import numpy as nm
from matplotlib.pyplot import scatter  


# In[44]:


plt.scatter(x_train,y_train)
plt.show()


# In[ ]:




