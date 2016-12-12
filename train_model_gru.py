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

char2id = pickle.load(open("dataset/char2id.p"))
id2char = pickle.load(open("dataset/id2char.p"))
maxSequenceLength = max_recipe_length + 1

print('Building training model...')
hiddenStateSize = 128
hiddenLayerSize = 128
hiddenLayerSize2 = 128

model = Sequential()
model.add(GRU(hiddenStateSize, return_sequences = True, input_shape=(maxSequenceLength, len(char2id))))
model.add(TimeDistributed(Dense(hiddenLayerSize)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dense(len(char2id))))  # Add another dense layer with the desired output size.
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001))

num_epochs = 10
batch_size = 128
model.fit_generator(training_set_generator(batch_size), int(num_recipes / batch_size), num_epochs, verbose=1)

model.save_weights('cocktail_weights_gru.h5')