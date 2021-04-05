import random

import numpy as np
from sklearn.metrics import r2_score

from linear_algebra import LinearAlgebra

la = LinearAlgebra()

class RegressionMixin:
	"""Calculate the score."""
	def r2_score(self,y_true,y_predicted):
		return r2_score(y_true,y_predicted)
	
	def mse_score(self,y_true,y_predicted):
		return np.mean((y_true - y_predicted)**2)


class StochasticBaseRegression:
    
	def __init__(self, learning_rate=0.01,n_iters=10, batch_size=100,decay_rate=0.70,tolerance=1e-17):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.tolerance = tolerance
		self.batch_size = batch_size
		self.decay_rate = decay_rate
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		data = np.array(list(zip(X,y)))
        # init parameters
		N_samples,n_features= X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		dw_diff = 0
		db_diff = 0

		for _ in range(self.n_iters):
			random.shuffle(data)

			for start in range(0,N_samples,self.batch_size):
				stop = start + self.batch_size
				x_batch,y_batch = data[start:stop,0],data[start:stop,1]
				#Shapes the array as (batch_size,n_features)
				x_batch,y_batch = np.array([elem.ravel() for elem in x_batch]),np.array([elem.ravel() for elem in y_batch])
				n_samples,_ = x_batch.shape

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

class MultiRegression(RegressionMixin,StochasticBaseRegression):

	def _approximation(self, X, w, b):
		return np.dot(X, w) + b

	def _predict(self, X, w, b):
		return np.dot(X, w) + b


