import math
import collections
from collections import Counter
import random

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from linear_algebra import LinearAlgebra
from statistics import Statistics


la = LinearAlgebra()
stats = Statistics()

random.seed(0)
y = np.random.randn(1000,1)
x = np.random.randn(1000,5)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

class RegressionMixin:
	"""Calculate the score."""
	def mse_score(self,y_true,y_predicted):
		return np.mean((y_true - y_predicted)**2)

class StochasticBaseRegression:
    
	def __init__(self, learning_rate=0.5,n_iters=100, batch_size=10,decay_rate=0.9,tolerance=1e-06):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.tolerance = tolerance
		self.batch_size = batch_size
		self.decay_rate = decay_rate
		self.weights = None
		self.bias = None

	def in_random_order(self,data):
		indexes = [i for i,_  in enumerate(data)]
		random.shuffle(indexes)
		for i in indexes:
			yield data[i]

	def fit(self, X, y):
		data = np.array(list(zip(X,y)))
        # init parameters
		n_samples,_= X.shape
		self.bias = 0
		dw_diff = 0
		db_diff = 0

		for _ in range(self.n_iters):
			random.shuffle(data)

			for start in range(0,n_samples,self.batch_size):
				stop = start + self.batch_size
				x_batch,y_batch = data[start:stop,0],data[start:stop,1]
				#Shapes the array as (batch_size,n_features)
				x_batch,y_batch = np.array([elem.ravel() for elem in x_batch]),np.array([elem.ravel() for elem in y_batch])
				_, n_features = x_batch.shape
				self.weights = np.zeros(n_features)

	        	# stochastic gradient descent
				y_predicted = self._approximation(x_batch, self.weights, self.bias)
				residual = la.vector_subtract(y_predicted,y_batch)
				# compute gradients
				dw = (1 / n_samples) * np.dot(x_batch.T,residual)
				db = (1 / n_samples) * np.sum(residual)
				dw_diff = self.decay_rate * dw_diff - self.lr * dw
				db_diff = self.decay_rate * db_diff - self.lr * db


				if np.all(np.abs(dw_diff)) <= self.tolerance:
					break

				# update parameters
				self.weights = la.vector_subtract(self.weights,(self.lr * dw))
				self.bias -= self.lr * db

	def predict(self, X):
		return self._predict(X, self.weights, self.bias)

	def _predict(self, X, w, b):
		raise NotImplementedError()

	def _approximation(self, X, w, b):
		raise NotImplementedError()

class LinearRegression(RegressionMixin,StochasticBaseRegression):

	def _approximation(self, X, w, b):
		return np.dot(X, w) + b

	def _predict(self, X, w, b):
		return np.dot(X, w) + b


lr = LinearRegression()
lr.fit(x_train,y_train)
print("WEIGHTS AND BIAS")
print(lr.weights,lr.bias)
y_pred = lr.predict(x_test)
mse_score_class = lr.mse_score(y_test,y_pred)
print("mse_SCORE_class: " + str(mse_score_class))

# data = np.array(list(zip(x_train,y_train)))
# x_batch = data[2:10,1]
# ravel = np.array([elem.ravel() for elem in x_batch])
# print(ravel.shape)

