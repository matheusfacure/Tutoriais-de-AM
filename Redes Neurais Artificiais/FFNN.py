import numpy as np
import pandas as pd
class FFNN(object):

	def __init__(self, n_hl):

		# tamanho da camada oculta
		self.n_hl = n_hl


	def logistica(self, z):
		return 1/(1+np.exp(-z)) 

	def dlogistic(self, z):
		'''Derivada da função logistica'''
		return np.exp(-z)/((1+np.exp(-z))**2)


	def forward(self, X):
		self.z2 = np.dot(X, self.W_in_to_hl)
		self.a2 = self.logistica(self.z2)
		self.z3 = np.dot(self.a2, self.W_hl_to_out)
		return self.sigmoid(self.z3) # yhat

	
	def costFunction(self, X, y):
		'''Erros quadrados'''
		self.yHat = self.forward(X)
		E = (1/2) * sum((y-self.yHat)**2)
		return E

	def dcostFunction(self, X, y):
		
		self.yHat = self.forward(X)

		delta3 = -(y-self.yHat) * self.dlogistic(self.z3)
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)  

		return dJdW1, dJdW2


	def fit(self, X_train, y_train):

		# inicializa os pesos
		self.W_in_to_hl = np.random.randn(X_train.shape[1], self.n_hl)
		self.W_hl_to_out = np.random.randn(self.n_hl, X_train.shape[1])




if __name__ == '__main__':

	X = np.array((np.random.normal(10, 1, 100),
				  np.random.normal(10, 1, 100),
				  np.random.normal(10, 1, 100)), dtype=float).T
	y = (X[:, 0] * X[:, 1]) / (X[:, 1] ** X[:, 0]) 

