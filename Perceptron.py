from sklearn.linear_model import Perceptron
x=[[1,0],[1,1],[2,0]]
y=["boy","girl","boy"]
ML=Perceptron()
ML=ML.fit(x,y)
result=ML.predict([[1,0],[1,1]])
print(result)
