#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
from tensorflow import keras
import numpy as np

print("TFversion:{}".format(tf.__version__))

dataset = np.genfromtxt('NR-ER-train/names_labels.csv', delimiter = ',', usecols = (1))
dataset = dataset.tolist()

dataset_test = np.genfromtxt('NR-ER-test/names_labels.csv', delimiter = ',', usecols = (1))
dataset_test = dataset_test.tolist()

Y = [1 if i == 1.0 else 0 for i in dataset ]
Y_test = [1 if i == 1.0 else 0 for i in dataset_test ]

Y = keras.utils.to_categorical(Y,num_classes=2)
Y_test = keras.utils.to_categorical(Y_test,num_classes=2)

X = np.load('NR-ER-train/names_onehots.npy') 
X = X.item()
X = X['onehots']

X_test = np.load('NR-ER-test/names_onehots.npy') 
X_test = X_test.item()
X_test = X_test['onehots']


X_train = X
Y_train = Y
#X_val = X[7001:]
#Y_val = Y[7001:]

print(X_train.shape)
#print(X_val.shape)
print(X_test.shape)
print(Y_train.shape)
#print(Y_val.shape)
print(Y_test.shape)

X_train = np.reshape(X_train, [7697, 1,72,398])
#X_val = np.reshape(X_val,[696, 1, 72, 398])
X_test = np.reshape(X_test,[265, 1, 72, 398])

print(X_train.shape)
#print(X_val.shape)
print(X_test.shape)
print(Y_train.shape)
#print(Y_val.shape)
print(Y_test.shape)

class_weight = {0:1, 1:6}


input_shape=(1, 72, 398)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu', input_shape=input_shape, padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size=(5,5), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding = 'same'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size = (1,1), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(364, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train, epochs=20, batch_size=25, validation_data = (X_test,Y_test), class_weight = class_weight)

model.save('./saved_weights')

print(history.history.keys())

check = model.evaluate(X_test,Y_test)
print(model.metrics_names)

print(check)
