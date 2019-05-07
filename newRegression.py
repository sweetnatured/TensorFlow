import pathlib

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path=keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names=['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Modelyear','Origin']

raw_data=pd.read_csv(data_path,names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
cdata=raw_data.copy()
cdata=cdata.dropna()

cdata.pop('Origin')
data_stat=cdata.describe()

train_data=cdata.sample(frac=0.8,random_state=0)
test_data=cdata.drop(train_data.index)

train_label=train_data.pop('MPG')
test_label=test_data.pop('MPG')

normed_train_data=train_data/train_data.max()
normed_test_data=test_data/test_data.max()

model=keras.Sequential([

    keras.layers.Dense(64,activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(64,activation=tf.nn.relu),
    keras.layers.Dense(1)

])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.RMSprop(0.001),
              metrics=['mean_absolute_error','mean_squared_error']
              )

model.fit(normed_train_data,train_label,epochs=10000)

prediction=model.predict(normed_test_data).flatten()

plt.scatter(test_label, prediction)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()