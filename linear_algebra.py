import math

class LinearAlgebra():

	def vector_add(self,v,w):
		return [v_i + w_i for v_i,w_i in zip(v,w)]
	
	def vector_subtract(self,v,w):
		return [v_i - w_i for v_i,w_i in zip(v,w)]
	
	def vector_sum(self,vectors):
		return reduce(vector_add,vectors)
	
	def scalar_multiply(self,c,v):
		return [c*v_i for v_i in v]
	
	def vector_mean(self,vectors):
		n = len(vectors)
		return self.scalar_multiply(1/n,self.vector_sum(vectors))
	
	def dot(self,v,w):
		return sum(v_i*w_i for v_i,w_i in zip(v,w))
	
	def sum_of_squares(self,v):
		return self.dot(v,v)
	
	def magnitude(self,v):
		return math.sqrt(self.sum_of_squares(v))
	
	def squared_distance(self,v,w):
		return self.sum_of_squares(vector_subtract)
	
	def distance(self,v,w):
		return self.magnitude(self.vector_subtract(v,w))
	
	def shape(self,A):
		num_rows = len(A)
		num_cols = len(A[0]) if A else 0
		return num_rows,num_cols
	
	def get_row(self,A,i):
		return A[i]
	
	def get_column(self,A,j):
		return [A_i[j] for A_i in A]
	
	def make_matrix(self,num_rows,num_cols,entry_fn):
		return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]
	
	def is_diagonal(self,i,j):
		return 1 if i == j else 0 