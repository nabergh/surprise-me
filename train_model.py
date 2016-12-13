import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU
from keras.optimizers import Nadam
from keras.layers.wrappers import TimeDistributed
from keras import callbacks
import pickle

from load_dataset import training_set_generator
from load_dataset import num_recipes
from load_dataset import max_recipe_length

char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))

max_sequence_length = max_recipe_length + 1

print('Building training model...')
hiddenStateSize = 128
hiddenLayerSize = 256 
hiddenLayerSize2 = 256

model = Sequential()

model.add(GRU(hiddenStateSize, return_sequences = True, input_shape=(max_sequence_length, len(char2id))))
model.add(TimeDistributed(Dense(hiddenLayerSize)))
model.add(TimeDistributed(Activation('relu')))

model.add(GRU(hiddenLayerSize, return_sequences = True, input_shape=(max_sequence_length, len(char2id))))
model.add(TimeDistributed(Dense(hiddenLayerSize2)))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Dense(len(char2id))))  # Add another dense layer with the desired output size.
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='categorical_crossentropy', optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-08, schedule_decay=0.004))

print(model.summary())

cb = []
cb.append(callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=False))

num_epochs = 10
batch_size = 128
samp_per_epoch = 30848 # multiple of 128 closest to num_recipes
with tf.device('/gpu:0'):
	hist = model.fit_generator(training_set_generator(batch_size), samp_per_epoch, num_epochs, verbose=1, callbacks=cb)
model.save_weights('cocktail_weights.h5')
print(hist.history)
