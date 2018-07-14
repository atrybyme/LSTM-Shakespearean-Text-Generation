##importing important libraries

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
import h5py
from keras.callbacks import ModelCheckpoint
##importing x_train y_train data
print("Loading Data . . .")
y_train = np.load("y_train.npy")
x_train = np.load("x_train.npy")
print("Data Loading Complete")
##input shape
in_shape = x_train.shape[1:]

##creating Sequential model
print("Building Model")
model  = Sequential()
model.add(LSTM(256,input_shape=in_shape,return_sequences=True))
model.add(Dropout(0.2))          
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print("Model Building Complete")

#for training we will create a mini batch from our dataset
mini_batch_size = 64
print("Training Started")
model.fit(x_train,y_train,epochs=19,verbose=1,batch_size=mini_batch_size,validation_split=0.05,callbacks=callbacks_list)
print("Training Complete, Saving Model.")
model.save("trained_model_1.h5")
print("Model Saved")




