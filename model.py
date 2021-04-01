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

np.random.seed(0)
np.set_printoptions(precision=1)

y = np.random.randn(100,1)
x = np.random.randn(100,2)

class StochasticGradientDescent:
	
	def __init__(self,lr):
		self.lr = lr
		self.weights = None
		self.bias = None

	def in_random_order(self,data):
		indexes = [i for i,_ in enumerate(data)]
		random.shuffle(indexes)

		for i in indexes:
			yield data[i]

	def gradient(self,x_i,y_i,y_predicted)		
		dw = (1 / n_samples) * np.dot(x_i.T, (y_predicted - y_i))
		db = (1 / n_samples) * np.sum(y_predicted - y_i)
		return dw,db

	def stochastic_descent(self,target_fn,gradient_fn,x,y):
		data = np.array(list(zip(x,y)))

		for x_i,y_i in self.in_random_order(data):
			y_predicted = target_fn(x_i, self.weights, self.bias)
			dw,db = self.gradient(x_i,y_i,y_predicted)

			self.weights -= self.lr * dw
			self.bias -= self.lr * db

	def negate(self,f):
		return lambda *args,**kwargs:-f(*args,**kwargs)
	
	def negate_all(self,f):
		return lambda *args,**kwargs:[-y for y in f(*args,**kwargs)]
	
	def maximize_stochastic(self,target_fn,gradient_fn,x,y):
		return self.minimize_stochastic(self.negate(target_fn),
										self.negate_all(gradient_fn),
										x,y)

	def predict(self, X):
		return self._predict(X, self.weights, self.bias)

	def _predict(self, X, w, b):
	    raise NotImplementedError()

	def _approximation(self, X, w, b):
	    raise NotImplementedError()

class BaseRegression(StochasticGradientDescent):

	def __init__(self):
		super.__init__()
	
	def error(self,a,b,x_i,y_i):
		return np.subtract(y_i,self._predict(a,b,x_i))
	
	def sum_of_squared_errors(self,a,b,x,y):
		return np.sum(self.error(a,b,x_i,y_i)**2 for x_i,y_i in zip(x,y))
	
	def total_sum_of_squares(self,y):
		np.sum([v**2 for v in stats.de_mean(y)])
	
	def r_squared(self,alpha,beta,x,y):
		return 1.0 - (np.divide(self.sum_of_squared_errors(alpha,beta,x,y),self.total_sum_of_squares(y)))
	
	def _predict(self, X, w, b):
	    raise NotImplementedError()

	def _approximation(self,a,b,x_i):
		raise NotImplementedError()


class MultipleRegression(BaseRegression):
		
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b

import numpy as np


