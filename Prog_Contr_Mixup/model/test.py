from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(123)
import tensorflow as tf
tf.random.set_seed(123)
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# import pdb; pdb.set_trace()
	trainX = trainX.reshape((trainX.shape[0], 784)) /255.0
	testX = testX.reshape((testX.shape[0],784)) / 255.0
	trainX = trainX[:6000,:]
	trainY = trainY[:6000]
	trX,tstX,trY,tstY = train_test_split(trainX,trainY,shuffle=True, test_size=0.1)
# 	testX /= 255.0
	# one hot encode target values
	trY = to_categorical(trY)
	tstY = to_categorical(tstY)
	return trX,trY,tstX,tstY #trainX, trainY, testX, testY
	
def define_model():
	model = Sequential()
	model.add(Dense(100, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(10,activation='softmax'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

trainX, trainY, testX, testY = load_dataset()
model = define_model()
history = model.fit(trainX, trainY, epochs=50, batch_size=128, verbose=1)
_, acc = model.evaluate(testX, testY, verbose=1)
print('> %.3f' % (acc * 100.0))