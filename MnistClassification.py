from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images= train_images / 255
test_images= test_images/255

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.tanh),
    keras.layers.Dense(128,activation=tf.nn.tanh),
    keras.layers.Dense(10,activation=tf.nn.softmax)

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']

              )

model.fit(train_images,train_labels,epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss,test_acc)


predictions = model.predict(test_images)

for i in range(100):

    print("Target:{}".format(np.argmax(predictions[i])),"Value:{}".format(test_labels[i]))