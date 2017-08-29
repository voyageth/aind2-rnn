import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size:i+window_size+1])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1))
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # 'a' = 97, 'z' = 122
    
    u_chars = set(text)
    print("before : " + str(u_chars))
    
    for u_char in u_chars:
        if u_char < 'a' and u_char not in punctuation:
            text = text.replace(u_char, ' ')
        if u_char > 'z' and u_char not in punctuation:
            text = text.replace(u_char, ' ')
            
    u_chars = set(text)
    print("after : " + str(u_chars))
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    pair_count = int((len(text) - window_size) / step_size)
    
    for i in range(pair_count):
        index = i * step_size
        inputs.append(text[index:index + window_size])
        outputs.append(text[index + window_size:index + window_size + 1])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
