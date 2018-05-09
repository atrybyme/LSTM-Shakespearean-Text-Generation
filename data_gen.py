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
