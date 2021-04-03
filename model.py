import math
import collections
from collections import Counter
import random

import pandas
import numpy as np

from linear_algebra import LinearAlgebra
from statistics import Statistics

la = LinearAlgebra()
stats = Statistics()


class BaseRegression:
    
	def __init__(self, learning_rate=0.001, n_iters=1000):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape

        # init parameters
		self.weights = np.zeros(n_features)
		self.bias = 0

        # gradient descent
		for _ in range(self.n_iters):
			y_predicted = self._approximation(X, self.weights, self.bias)
			residual = np.array(la.vector_subtract(y_predicted,y))

			# compute gradients
			dw = (1 / n_samples) * np.dot(X.T,residual)
			dw_list.append(dw)
			db = (1 / n_samples) * np.sum(residual)
			db_list.append(db)

			# update parameters
			self.weights = la.vector_subtract(self.weights,(self.lr * dw))
			self.bias -= self.lr * db

	def predict(self, X):
		return self._predict(X, self.weights, self.bias)

	def _predict(self, X, w, b):
		raise NotImplementedError()

	def _approximation(self, X, w, b):
		raise NotImplementedError()

class LinearRegression(BaseRegression):

	def _approximation(self, X, w, b):
		return np.dot(X, w) + b

	def _predict(self, X, w, b):
		return np.dot(X, w) + b
