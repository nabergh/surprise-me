import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras import callbacks
import pickle
import time

from load_dataset import training_set_generator
from load_dataset import num_recipes
from load_dataset import max_recipe_length

char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))

hiddenStateSize = 128
hiddenLayerSize = 128
max_sequence_length = max_recipe_length + 1

print('Building training model...')
hiddenStateSize = 128
hiddenLayerSize = 128
model = Sequential()

model.add(LSTM(hiddenStateSize, return_sequences = True, input_shape=(max_sequence_length, len(char2id))))
model.add(TimeDistributed(Dense(hiddenLayerSize)))
model.add(TimeDistributed(Activation('relu'))) 
model.add(TimeDistributed(Dense(len(char2id))))
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001))

print(model.summary())
cb = []
cb.append(callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=False))

t0 = time.time()
num_epochs = 1000
batch_size = 128 
with tf.device('/gpu:0'):
	model.fit_generator(training_set_generator(batch_size), int(num_recipes / batch_size), num_epochs, verbose=1, callbacks = cb)
model.save_weights('cocktail_weights.h5')
print("Time elapsed to train model for " + str(num_epochs) + " epochs: " + str(time.time() - t0))
