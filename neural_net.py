#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy, pandas
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt

# class AccuracyHistory(keras.callbacks.Callback):
# 	def on_train_begin(self, logs={}):
# 		self.acc = []
#
# 		def on_epoch_end(self, batch, logs={}):
# 			self.acc.append(logs.get('acc'))

# load dataset
inframe = pandas.read_csv("indata.csv", header=None)
indataset = inframe.values

outframe = pandas.read_csv("outdata.csv", header=None)
outdataset = outframe.values

print(indataset.shape)
plt.waitforbuttonpress()

print("done loading csv")
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(900, kernel_size=(3, 3), strides=(1, 1),
	             activation='relu',
	             input_shape=(30, 30, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(500, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# model.fit(indataset, outdataset,
#           batch_size=64,
#           epochs=10,
#           verbose=1,
#           # validation_data=(x_test, y_test),
#           # callbacks=[history])


# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
#
# history = AccuracyHistory()
# plt.plot(range(1,11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=64, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, indataset, outdataset, cv=kfold)
print(results)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
