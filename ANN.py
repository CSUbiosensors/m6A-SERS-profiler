from sklearn.utils import shuffle
import pandas as pd
train=pd.read_csv('./Nucleotides.csv', engine='python')
train=shuffle(train, random_state=1)

from sklearn.preprocessing import StandardScaler
def prepare_x(train):
    ndarray_train=train.values
    features=ndarray_train[:,1:]
    labels=ndarray_train[:,0]
    scaler=StandardScaler().fit(features)
    norm_features=scaler.transform(features)
    return norm_features, labels

import numpy as np
X, Y=prepare_x(train)

nb_features = 1015 
X_train = np.zeros((len(X), nb_features, 1))
X_train[:, :, 0] = X[:, :nb_features]

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.05, random_state=1)
for train_index, test_index in sss.split(X_train, Y):
    X_t, X_testing = X_train[train_index], X_train[test_index]
    Y_t, Y_testing = Y[train_index], Y[test_index]

from sklearn.model_selection import KFold
kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(X_t):
    X_train, X_test = X_t[train_index], X_t[test_index]
    Y_train, Y_test = Y_t[train_index], Y_t[test_index]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers, regularizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPool1D, BatchNormalization, ReLU
tf.random.set_seed(3)

model = Sequential()
model.add(Flatten(input_shape=(1015, 1)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.summary()

import os
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
Adam=Adam(lr=0.00001)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
checkpoint_save_path = "./checkpoint/No20240321.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('- - - - - - - load the model- - - - - - -')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True, save_best_only=True)
history = model.fit(X_train, Y_train, epochs=500, validation_data=(X_test, Y_test), batch_size=209, callbacks=[cp_callback])

history_df = pd.DataFrame(history.history)
species_prediction=model.predict(X_testing)
