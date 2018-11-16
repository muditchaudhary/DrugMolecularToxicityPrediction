import tensorflow as tf
from tensorflow import keras

import numpy as np

#dataset_test = np.genfromtxt('NR-ER-test/names_labels.csv', delimiter = ',', usecols = (1))
#Y_test = [1 if i == 1.0 else 0 for i in dataset_test ]
#Y_test = keras.utils.to_categorical(Y_test,num_classes=2)

X_test = np.load('./NR-ER-test/names_onehots.npy') 
X_test = X_test.item()
X_test = X_test['onehots']
X_test = np.reshape(X_test,[len(X_test), 1, 72, 398])

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

print(model.summary())

model.load_weights('./saved_weights')

predictions = model.predict(X_test)

f = open('labels.txt', 'w')



for i in range(len(predictions)):
    f.write(str(np.argmax(predictions[i]))+"\n")
