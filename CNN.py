from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import backend as K
from keras.models import load_model

import tensorflow as tf
import os
import numpy as np


def to_categorical(labels, num_classes):
	return keras.utils.to_categorical(labels, num_classes)
	
def create_model(input_shape):
	model = Sequential()
	model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(22))
	model.add(Activation('softmax'))

	print(model.summary())
	return model

def train_model(model, xtrain, ytrain, xtest, ytest, batch_size, epochs):

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',                                                                                                                            
              optimizer=opt,
              metrics=['accuracy'])

	print('Not using data augmentation.')
	model.fit(xtrain, ytrain,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(xtest, ytest),
			shuffle=True)

	return model

def evaluate_model(model, xtest, ytest):
	# Score trained model.
	scores = model.evaluate(xtest, ytest, verbose=1)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])


def save_model(model, save_dir, model_name):
	# Save model and weights
	if not os.path.isdir(save_dir):
	    os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.model.save(model_path)
	print('Saved trained model at %s ' % model_path)

