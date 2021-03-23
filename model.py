import pandas
import np
import math
import collections
from collections import Counter

class Statistics():

	def mean(self,x):
		return sum(x)/len(x)

	def median(self,v):
		n = len(v)
		sorted_v = sorted(v)
		midpoint = n//2

		if n%2 == 1:
			return sorted_v[midpoint]
		else:
			low_point = midpoint-1
			high_point = midpoint
			return (sorted_v[low_point] + sorted_v[high_point])/2

	def quantile(self,x,p):
		p_index = int(p*len(x))
		return sorted(x)[p_index]

	def mode(self,x):
		return collections.Counter(x).most_common()

	def range(self,x):
		return max(x) - min(x)

	def de_mean(self,x):
		x_bar = self.mean(x)
		return [x_i - x_bar for x_i in x]

	def variance(self,x):
		n = len(x)
		deviations = self.de_mean(x)
		return la.sum_of_squares(deviations)/(n-1)

	def standard_deviation(self,x):
		return math.sqrt(self.variance(x))

	def interquantile_range(self,x):
		return self.quantile(x,0.75) - self.quantile(x,0.25)

	def covariance(self,x,y):
		n = len(x)
		if n != len(y):
			print("x and y vectors must have the same length!")
		return la.dot(self.de_mean(x),self.de_mean(y))/(n-1)
	
	def correlation(self,x,y):
		stdev_x = self.standard_deviation(x)
		stdev_y = self.standard_deviation(y)
		if stdev_x > 0 and stdev_y > 0:
			return self.covariance(x,y)/(stdev_x*stdev_y)
		return 0

stats = Statistics()

class LinearRegression():

	def predict(self,a,b,x_i):
		return np.sum([np.multiply(b,x_i),a])

	def error(self,a,b,x_i,y_i):
		return np.subtract(y_i,self.predict(a,b,x_i))

	def sum_of_squared_errors(self,a,b,x,y):
		return np.sum(self.error(a,b,x_i,y_i)**2 for x_i,y_i in zip(x,y))

	def least_square_fit(self,x,y):
		beta = np.multiply(stats.correlation(x,y),stats(np.divide(standard_deviation(y),stats.standard_deviation(x))))
		alpha = np.subtract(stats.mean(y),np.multiply(beta,stats.mean(x)))
		return alpha,beta

	def total_sum_of_squares(self,y):
		np.sum([v**2 for v in stats.de_mean(y)])

	def r_squared(self,alpha,beta,x,y):
		return 1.0 - (np.divide(self.sum_of_squared_errors(alpha,beta,x,y),self.total_sum_of_squares(y)))

	def squared_error(self,x_i,y_i,theta):
		alpha,beta = theta
		return self.error(alpha,beta,x_i,y_i)**2

	def squared_error_gradient(x_i,y_i)
		pass


