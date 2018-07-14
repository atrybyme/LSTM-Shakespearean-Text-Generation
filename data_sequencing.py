##importing important libraries
import numpy as np

##load one hot encoded data
one_hot_encoded_data = np.load("character_Data_onehotencoded.npy")

##decide the length of sequence
seq_len = 140
y_train = []
##we now have to devide the data into sequence of length 140 each
##these sequence will act as our final dataset.
no_of_dataset_elements = (one_hot_encoded_data.shape)[0] - seq_len
print("Total number of files in dataset is : ",no_of_dataset_elements,"\n")
for i in range(no_of_dataset_elements):
    a = "./seq_data/sequential_data_" + str(i) + ".npy"
    x_train = one_hot_encoded_data[i:i+seq_len]
    y_train.append(one_hot_encoded_data[i+seq_len])
    np.save(a,x_train)
    if(i%1000==0):
        print("Sequencing completed : ",int((i/no_of_dataset_elements)*100))
##saving y_train
np.save("y_train.npy",np.asarray(y_train))

