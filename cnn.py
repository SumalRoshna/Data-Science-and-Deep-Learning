import tensorflow as tf
from tensorflow import keras
mnistDB=keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnistDB.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[5], cmap='binary')

x_train=x_train.reshape((60000,28,28,1))
x_test= x_test.reshape((10000,28,28,1))
x_train=x_train.astype('float32')/255
x_test= x_test.astype('float32')/255

cnnModel_MNIST=keras.models.Sequential()

cnnModel_MNIST.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=x_train.shape[1:]))

cnnModel_MNIST.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
cnnModel_MNIST.add(keras.layers.MaxPooling2D((2,2)))
cnnModel_MNIST.add(keras.layers.Dropout(0.25))
cnnModel_MNIST.add(keras.layers.Flatten())
cnnModel_MNIST.add(keras.layers.Dense(128, activation='relu'))
cnnModel_MNIST.add(keras.layers.Dropout(0.25))
cnnModel_MNIST.add(keras.layers.Dense(10, activation='softmax'))
cnnModel_MNIST.summary()

cnnModel_MNIST.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

cnnModel_MNIST.fit(x_train,y_train,epochs=1,batch_size=10)
test_loss, test_accuracy= cnnModel_MNIST.evaluate(x_test, y_test)

print(test_loss, test_accuracy)