# Shakespearean Text Generation using LSTM.
This repository contains step by step code for generation of __Text that resemble Shakespeare Writings__.
## Long Short Term Memory Networks(LSTMs)
With the discovery of Recurrent Neural Networks(RNNs) , a whole new dimension was added to the study of Deep Learning. The dimension of Time. With RNNs aur models upgraded from images to videos, time stationary signals to voices, etc.
But earlier model of RNNs suffered from a major problem with Gradients.As the model backpropogates through time gradients either vanishes or explodes depending wither the max gradient value was less than one or larger than one.
LSTMs try to solve this problem using 4 gates.
Basic Structure of LSTM is as follows
-[LSTM][1]
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
The first file that we need to run is [one_hot_encoded.py][3]. This program loads the text file and do the following things:
-
>
[1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png
