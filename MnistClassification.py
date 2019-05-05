from __future__ import absolute_import, division, print_function

import tensorflow as tf                             #İMPORT THE LİBRİARİES
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist               #GET THE DATASET FROM KERAS DATABES
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    #APPOİNT THE LABES FROM DATABASE

train_images= train_images / 255      #MAKE DATA NORMALIZATION
test_images= test_images/255

model = keras.Sequential([                    #CREATE THE NETWORK STRUCTURE (28*28--128--128--10)

    keras.layers.Flatten(input_shape=(28,28)),                  #FLATTEN MAKES ROW DATA FROM İNPUT          
    keras.layers.Dense(128,activation=tf.nn.tanh),              #CREATES THE FİRST LAYER THAT HAS 128 NEURON AND TANH ACT. FUNCTİON     
    keras.layers.Dense(128,activation=tf.nn.tanh),              #CREATES THE SECOND LAYER THAT HAS 128 NEURON AND TANH ACT. FUNCTİON
    keras.layers.Dense(10,activation=tf.nn.softmax)             #CREATES THE OUTPUT LAYER THAT HAS SOFTMAX FUNCTİON TO CALCULATE PROBABİLY DİSTRUBİTİON

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']

              )

model.fit(train_images,train_labels,epochs=10)                        #TRAİNS THE DATA TİMES OF EPOCHS NUMBER

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss,test_acc)


predictions = model.predict(test_images)               # PREDİCTS DATA FROM NETWORK

for i in range(100):

    print("Target:{}".format(np.argmax(predictions[i])),"Value:{}".format(test_labels[i]))      #DOES DATA VALİDATİON FOR COMPARE RESULTS
