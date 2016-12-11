import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import pickle

from load_dataset import training_set_generator
from load_dataset import num_recipes

char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))

hiddenStateSize = 128
hiddenLayerSize = 128
max_recipe_length = 500

print('Building training model...')
hiddenStateSize = 128
hiddenLayerSize = 128
model = Sequential()

model.add(LSTM(hiddenStateSize, return_sequences = True, input_shape=(max_recipe_length, len(char2id))))
model.add(TimeDistributed(Dense(hiddenLayerSize)))
model.add(TimeDistributed(Activation('relu'))) 
model.add(TimeDistributed(Dense(len(char2id))))
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001))

print(model.summary())

num_epochs = 10
model.fit(inputChars, nextChars, batch_size = 128, nb_epoch = 10)
model.fit_generator(training_set_generator, num_recipes, num_epochs, verbose=1)

model.save_weights('cocktail_weights.h5')