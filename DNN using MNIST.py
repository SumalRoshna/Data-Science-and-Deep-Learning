# -*- coding: utf-8 -*-
"""deeplearning_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XZtcnqULrjKWogsKSqPazOpNKD9Db61E
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
mnistDB= keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnistDB.load_data()
x_train= x_train.reshape((60000,28*28))
x_test= x_test.reshape((10000,28*28))
x_train=x_train.astype('float32')/255
x_test= x_test.astype('float32')/255

ml=keras.models.Sequential()
ml.add(keras.layers.Dense(units=50, activation='relu'))
ml.add(keras.layers.Dense(units=100, activation='relu'))
ml.add(keras.layers.Dense(units=70, activation='relu'))
ml.add(keras.layers.Dense(units=10, activation='softmax'))

ml.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ml.fit(x_train, y_train, epochs=1, batch_size=16)
test_loss, test_accuracy=ml.evaluate(x_test,y_test)

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

img=load_img('/content/4.png', grayscale=True, target_size=(28,28))
plt.imshow(img)
img=img_to_array(img)
img=img.reshape(1,28*28)
img=img.astype('float32')/255.0
digit=ml.predict_step(img)
print('Digit is: ',digit[0])