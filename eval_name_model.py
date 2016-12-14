import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, LSTM, GRU
from keras.layers.core import Reshape
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import pickle

from load_dataset import training_set_generator
from load_dataset import num_recipes
from load_dataset import max_recipe_length

char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))

max_sequence_length = max_recipe_length + 1

print('Building Inference model...')

hiddenStateSize = 512
hiddenStateSize2 = 256
hiddenStateSize3 = 128
hiddenLayerSize = 128

recipe_node = Sequential()

recipe_node.add(GRU(hiddenStateSize, batch_input_shape=(max_sequence_length, len(char2id))))
recipe_node.add(Dense(hiddenStateSize3))
recipe_node.add(Activation('relu'))

title_node = Sequential()

title_node.add(GRU(hiddenStateSize2, input_shape=(1, 1, len(char2id)), stateful = True)
title_node.add(Dense(hiddenStateSize3)))
title_node.add(Activation('relu'))


inference_model = Sequential()
inference_model.add(Merge([recipe_node, title_node], mode='concat', concat_axis=-1))

inference_model.add(Reshape((1, hiddenStateSize3)))

inference_model.add(GRU(hiddenStateSize3, stateful = True))
inference_model.add(Dense(hiddenLayerSize))
inference_model.add(Activation('relu'))

inference_model.add(Dense(len(char2id)))
inference_model.add(Activation('softmax'))

print(inference_model.summary())


inference_model.load_weights('cocktail_weights.h5')
num_samples = 20
gen = name_training_set_generator(num_samples)

def printEmbedding(recipe):
    charIds = np.zeros(recipe.shape[0])
    for (idx, elem) in enumerate(recipe):
        charIds[idx] = np.nonzero(elem)[0].squeeze()
    print(''.join([id2char[x] for x in charIds]))


for i in range(0, num_samples):
    inference_model.reset_states()
    inputs, realName = gen(next)
    print("Real recipe:")
    printEmbedding(inputs[0])
    print("Real title:")
    printEmbedding(inputs[1])
    print("Generated titled:")
    
    startChar = np.zeros((1, 1, len(char2id)))
    startChar[0, 0, char2id['S']] = 1
    end = False
    sent = ""
    for i in range(0, max_sequence_length):
        nextCharProbs = inference_model.predict([inputs[0], startChar])

        nextCharProbs = np.asarray(nextCharProbs).astype('float64')
        nextCharProbs = nextCharProbs / nextCharProbs.sum()

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
