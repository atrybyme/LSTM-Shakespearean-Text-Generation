# Shakespearean Text Generation using LSTM.
This repository contains step by step code for generation of __Text that resemble Shakespeare Writings__.
## Long Short Term Memory Networks(LSTMs)
With the discovery of Recurrent Neural Networks(RNNs) , a whole new dimension was added to the study of Deep Learning. The dimension of Time. With RNNs aur models upgraded from images to videos, time stationary signals to voices, etc.
But earlier model of RNNs suffered from a major problem with Gradients.As the model backpropogates through time gradients either vanished or exploded depending wether the max gradient value was less than one or larger than one.
LSTMs try to solve this problem using 4 gates.
Basic Structure of LSTM is as follows
![LSTM][1]
__These gates are :__
- Input Gate(Sigmoid) : Controls wether to write a cell or not.
- Forget Gate(Sigmoid) : Controls Wether to erase data on the cell or not.
- Output Gate(Sigmoid) : Controls how much to reveal the cell for update.
- Gate Gate(Tanh) : Controls the write value of cell.
Here I use LSTMs to generate text.
## Some of the snippets from my Output are :
> that I have the count rousillon .I have a seem to me the moon , and the moon.

>Tongue .I will not be the count of the stores ,and the starries ,the singer ,and the starries.

>
## Requirements
- Python(3.5.2/2.7.14)
- Keras(2.0.8)
- Tensorflow-gpu(1.3.0)
- Pickle
- Numpy

_The entire model was trained on my laptop with Nvidia Geforce GTX 960M Graphics Processor._

## Procedure : 
### Step 1 : 
The repository contains [shakespeare.txt][2] which will act as our dataset. 
The first file that we need to run is [one_hot_encoded.py][3]. This program converts all the text from [shakespeare.txt][2] to lowercase and we take the amount of data we like for training.(__I took 100K characters for training__).Then it creates an array of unique characters and do one hot encoding. Once the whole data is encoded we save all our arrays as npy file and dictionary as pickle file for converting our onehot encoded data to characters.
>python one_hot_encoded.py
### Step 2 :
Then we need to run [data_sequencing.py][4].This file takes character_Data_onehotencoded(which will be created after 1st program) and creates various 2d array by accumulating sequence of characters that you want(__I took 140 as my sequence Length__).Each sequence acts as training input element(x) and the next element after the sequence acts as our output(y).The program save file y_train and all the sequential data in [seq_data][5].
_We are simply taking 140 character and then trying to predict 141th character_
>python data_sequencing.py
### Step 3 :
We then need to run [merger.py][8]. This files simply merges all the sequences to 1 big x_train file.
>python merger.py
### Step 4 :
Now we start our training by running [training.py][6]. This files load x_train and y_train and creates our LSTM model.
The model is as follows : 
- An Input LSTM layer having 264 filters.
- Another LSTM having 128 filters.
- A fully connected layer with Sigmoid Activation which acts as our output layer.

The programs also saves the model weights after each epoch if the accuracy increases.
> python training.py
### Step 5 : 
The final code we need to run is [prediction.py][7] This will create a dummy model same as the ones in made during traninng and load one of our weights. Then it randomly takes 140 character from our dataset and creating out sequences.

__Viola! We have our Shakespeare writing Dumb and Senseless things.__ 


#### I used one of the many configuration possible. The model contains many flaws I have discussed below along with some recommendations.You can change the following parameters to get better results :
- Take more data for training. I used only 100k characters you can take more than that for better result.
- I trained only 15 epochs due to hardware limitation. LSTMs are hard to train and converges slowly.If you want you increase the epochs for better result._But always have a look on validation data to avoid over fitting._
- Play with the modeL .I used a very simple LSTM model you can go for more complex model.(But keep in mind to change in the [prediction.py][7] same as you do in [training.py][6]).

[1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png
[2]: shakespeare.txt
[3]: one_hot_encoded.py
[4]: data_sequencing.py
[5]: seq_data/
[6]: training.py
[7]: prediction.py
[8]: merger.py
