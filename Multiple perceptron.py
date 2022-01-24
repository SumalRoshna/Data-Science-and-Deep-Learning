from sklearn.neural_network import MLPClassifier
x=[[1,0],[1,1],[2,0]]
y=["boy","girl","boy"]
ML=MLPClassifier(hidden_layer_sizes=(10),activation='relu')
ML=ML.fit(x,y)
result=ML.predict([[1,0],[1,1]])
print(result)
