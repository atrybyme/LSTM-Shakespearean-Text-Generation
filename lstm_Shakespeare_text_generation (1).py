
# coding: utf-8

# In[1]:


## import important libraries
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import h5py
import pickle


# In[2]:


## import shakespaere text coupus file convert all alphabets to lower case
dataset_name = "shakespeare.txt"
raw_data= open(dataset_name,"r").read().lower()


# In[3]:


## convert string data into numpy array of characters
raw_data_array = np.array(list(raw_data))

##clipping dataset because my laptop cannot preprocess this much data(lol)
raw_data_array = raw_data_array[0:200000]
##get all the uniqure characters in the array
unique_characters = np.unique(raw_data_array)
##save unique character to use later in future for prediction
np.save("unique_characters.npy",unique_characters)
char_to_int = dict((c,i) for i, c in enumerate(unique_characters))
int_to_char = dict((i,c) for i, c in enumerate(unique_characters))
##save int_to_char and char_to_int
pickle_out = open("char_to_int.pickle","wb")
pickle.dump(char_to_int, pickle_out)
pickle_out.close()
pickle_out = open("int_to_char.pickle","wb")
pickle.dump(int_to_char, pickle_out)
pickle_out.close()
##integer encoding the data
integer_encoded_data = [char_to_int[char] for char in raw_data_array]
## one_hot_encoding of the integer_encoded_dataset
one_hot_encoded_data = to_categorical(integer_encoded_data)
np.save("character_Data_onehotencoded_200000.npy",one_hot_encoded_data)


# In[6]:


##taking 140 previous character to get next character
##equal to the limit to tweet allowed,because even people dont remember their previous tweet
memory_length = 140
##convert dataset into memory sequence of length 140.
##Our LSTM model wil input of one such sequnce at once to output the next character.
x_data = [np.zeros((memory_length,len(unique_characters)))]
y_data = [np.zeros((len(unique_characters)))]
for i in range(0,len(raw_data_array)-memory_length):
    x_data = np.append(x_data,[one_hot_encoded_data[i:i+memory_length]],axis=0)
    y_data = np.append(y_data,[one_hot_encoded_data[i+memory_length]],axis=0)
    if (i%500 == 0):
        print("Data Encoded: ",(i/(len(raw_data_array)-memory_length))*100,"%")
x_data = x_data[1:]
y_data = y_data[1:]


# In[7]:


##Create LSTM Model
model  = Sequential()
model.add(LSTM(256,input_shape=x_data.shape[1:],return_sequences=True))
model.add(Dropout(0.2))          
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y_data.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')


# In[85]:



filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_data, y_data, epochs=40,verbose=1, batch_size=128,callbacks=callbacks_list)


# In[32]:





# In[66]:


##prediction time
##we will provide a 140 character long input to predict the next character.
##we will take next 1000 characters at output
indices_of_word = np.random.randint(0,high=10000,size=memory_length)
x_in = [np.zeros((len(unique_characters)))]
for i in range(memory_length):
    x_in = np.append(x_in,[one_hot_encoded_data[indices_of_word[i]]],axis=0)
x_in = x_in[1:]


# In[67]:


prediction = []
for i in range(1000):
    x_in_shaped = np.reshape(x_in,(1,x_in.shape[0],x_in.shape[1]))
    out = model.predict(x_in_shaped,verbose=0)
    x_in = np.append(x_in,out,axis=0)
    x_in = x_in[1:]
    prediction.append(int_to_char[np.argmax(out)])


# In[69]:


output = ''.join(prediction)


# In[70]:


print("our predicted output")
print(output)

