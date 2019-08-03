# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:50:05 2019

@author: Connor
"""

import pandas as pd
import numpy as np
from skimage import io, filters
import skimage as ski
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Sequential

np.random.seed(1)

#creates a fold by loading 5 datasets for training, and shuffling them, then
#loading one dataset for testing
def create_fold():
    x1 = np.load('./numpy_arrays/data_user2.npy')
    x2 = np.load('./numpy_arrays/data_kaggle.npy')
    x3 = np.load('./numpy_arrays/data_user1.npy')
    x4 = np.load('./numpy_arrays/data_user3.npy')
    x5 = np.load('./numpy_arrays/data_user4.npy')
    
    y1 = np.load('./numpy_arrays/labels_user2.npy')
    y2 = np.load('./numpy_arrays/labels_kaggle.npy')
    y3 = np.load('./numpy_arrays/labels_user1.npy')
    y4 = np.load('./numpy_arrays/labels_user3.npy')
    y5 = np.load('./numpy_arrays/labels_user4.npy')
    
    x_train = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    y_train = np.concatenate((y1, y2, y3, y4, y5), axis=0)
    
    x_train2, y_train2 = shuffle(x_train, y_train)
    
    
    x_test = np.load('./numpy_arrays/data_user5.npy')
    y_test = np.load('./numpy_arrays/labels_user5.npy')
    
    return x_train2, x_test, y_train2, y_test

save_structure=True
save_history=True
save_path = 'my_model_3.h5'
history_path = '/history_my_model_3'

x = np.load('./numpy_arrays/data_32_shuffle.npy')
y = np.load('./numpy_arrays/labels_32_shuffle.npy')-1

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    test_size=0.2)

#one hot encoding
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#begin to build network
model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(24))
model.add(Activation('sigmoid'))

optimizerAda = keras.optimizers.Adagrad(lr=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizerAda,
              metrics=['accuracy'])

print(model.summary())
history = model.fit(x_train, y_train_cat, batch_size=30, epochs=25, validation_split=0.2)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test_cat, y_pred)
print("Accuracy", acc)

#model.save('my_model_2.h5')
if(save_structure):
    model.save(save_path)
    
if(save_history):
    with open(history_path, 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)