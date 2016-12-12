import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, GRU
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import pickle

from load_dataset import training_set_generator
from load_dataset import num_recipes
from load_dataset import max_recipe_length


char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))

hiddenStateSize = 128
hiddenLayerSize = 128
max_sequence_length = max_recipe_length + 1


inference_model = Sequential()
inference_model.add(GRU(hiddenStateSize, batch_input_shape=(1, 1, len(char2id)), stateful = True))
inference_model.add(Dense(hiddenLayerSize))
inference_model.add(Activation('relu'))
inference_model.add(Dense(len(char2id)))
inference_model.add(Activation('softmax'))

inference_model.load_weights('cocktail_weights_gru.h5')


for i in range(0, 10):
    inference_model.reset_states()  # This makes sure the initial hidden state is cleared every time.
    startChar = np.zeros((1, 1, len(char2id)))
    startChar[0, 0, char2id['S']] = 1
    end = False
    sent = ""
    for i in range(0, max_sequence_length):
        nextCharProbs = inference_model.predict(startChar)

        # In theory I should be able to input nextCharProbs to np.random.multinomial.
        nextCharProbs = np.asarray(nextCharProbs).astype('float64') # Weird type cast issues if not doing this.
        nextCharProbs = nextCharProbs / nextCharProbs.sum()  # Re-normalize for float64 to make exactly 1.0.

        nextCharId = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()
        if id2char[nextCharId] == 'E':
            if not end:
                print("~~~~~")
            end = True
        else:
            sent = sent + id2char[nextCharId] # The comma at the end avoids printing a return line character.
        startChar.fill(0)
        startChar[0, 0, nextCharId] = 1
    print(sent)