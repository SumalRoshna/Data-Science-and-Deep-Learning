from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

bc.load_breast_cancer()
y=bc.target
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30)

knn= KNeighborsClassifier(n_neighbors=6)
knn_fit(x_train, y_train)
y_pred= knn.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplot(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=true, linewidth=2, linecolor="white", fmt="of", ax=ax)
plt.show()
