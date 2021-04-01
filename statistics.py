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