import numpy as np

merged_data_set  = []
seq_len = 140
one_hot_encoded_data = np.load("character_Data_onehotencoded.npy")
no_of_data_elements =one_hot_encoded_data.shape[0]- seq_len
for i in range(no_of_data_elements):
    x_data = np.load("./seq_data/sequential_data_" +str(i)+".npy")
    merged_data_set.append(x_data)
    if(i%100==0):
        print("Sequence merging : " ,i)
np.save("x_train.npy",np.asarray(merged_data_set))
