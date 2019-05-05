from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras

import numpy as np


imdb=keras.datasets.imdb                                                                          #GETS THE DATA
(train_data, train_labels),(test_data, test_labels)  = imdb.load_data(num_words=10000)            # TAKES THE MOST FREQUENTLY 10000 WORD


train_data=keras.preprocessing.sequence.pad_sequences(train_data,padding='post',maxlen=256)        #DOES DATA PREPROCESSİNG
test_data= keras.preprocessing.sequence.pad_sequences(test_data,padding='post',maxlen=256)

model= keras.Sequential()                                                                       #CREATES SEQUENTİAL NEUROL NETWORK
model.add(keras.layers.Embedding(10000,16))                                                     # EMBEDS THE 10000 WORDS TO 16 DİMENSİON VECTOR SO THAT İNCREASE QUALİTY OF NETWORK
model.add(keras.layers.GlobalAveragePooling1D())                                                #DOES POOLİNG
model.add(keras.layers.Dense(16,activation=tf.nn.relu))                                         #HİDDEN LAYER THAT USES RELU ACTİVATİON FUNCTİON
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))                                       #OUTPUT LAYER THAT USES SİGMOİD FUNCTİON

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])


train=model.fit(train_data,train_labels,epochs=40,batch_size=512)                                          #TRAİNS THE DATA

prediction=model.predict(test_data)

for i in range(100):                                                                                               #TESTS THE DATA
    print("target :{}".format(prediction[i]),"result: {}".format(test_labels[i]))