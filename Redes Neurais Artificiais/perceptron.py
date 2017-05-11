import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


class perceptron(object):
	
	def __init__(self, lrate = 1, w=None, maxiter=3, tol = 0, plot_= False):
		self.lrate = lrate
		self.maxiter = maxiter
		self.tol = tol
		self.plot_ = plot_
		self.w = w

	def __update_w(self, X_train, y_train):

		for xi, yi in zip(X_train, y_train):
			
			if int(np.sign(np.dot(xi, self.w.T))) != int(yi):
				self.w += (xi.T * self.lrate)*yi
			

	def __eval_perceptron(self, X_train, y_train):

		error = 0
		miss_indx = []
		for xi, yi, i in zip(X_train, y_train, range(len(y_train))):

			if int(np.sign(np.dot(xi, self.w.T))) != int(yi):
				error += 1

		return error


	def _plot(self, X, y, true_w=None):
		a, b = -self.w[1]/self.w[2], -self.w[0]/self.w[2] 
		l = np.linspace(-1,1)
		plt.plot(l, a*l+b, 'green')
		cols = {1: 'r', -1: 'b'}

		for x,s in zip(X, y):
			plt.plot(x[0], x[1], cols[s]+'o')

		if not true_w is None:
			a, b = -true_w[1]/true_w[2], -true_w[0]/true_w[2] 
			plt.plot(l, a*l+b, '-k')

		plt.show()
		plt.clf()


	def fit(self, X_train, y_train):
		X_train = np.array(X_train)
		X_train = np.insert(X_train, 0, 1, 1) # adiciona vies
		y_train = np.array(y_train)

		if self.w is None:
			self.w = np.random.normal(0, 10, len(X_train[0]))

		count = 0
		while True:

			error = self.__eval_perceptron(X_train, y_train)
			self.__update_w(X_train, y_train)

			count += 1
			if error < self.tol or count > self.maxiter:
				break

			if self.plot_:
				self._plot(X_train[:, 1:], y_train)


	def predict(self, X_test):

		X_test = np.insert(X_test, 0, 1, 1)
		return np.sign(np.dot(X_test, self.w.T))


			


if __name__ == '__main__':
	

	# gera dados linearmente separáveis em 2D
	x1,y1,x2,y2 = [np.random.uniform(-1, 1) for i in range(4)] # define 2 pontos
	w_target = np.array([x2*y1-x1*y2, y2-y1, x1-x2]) # gera vetor
	a, b = -w_target[1]/w_target[2], -w_target[0]/w_target[2] # para desenhar

	X = np.random.uniform(-1, 1, (100, 2)) # gera 100 pontos 
	y = np.sign(np.dot(np.insert(X, 0, 1, 1), w_target.T)) # gera targets

	
	# usa perceptron para separar os dados	
	clf = perceptron(maxiter=10, lrate = 0.1, plot_ = True)
	clf.fit(X, y)
	pred = clf.predict(X)
	print(sum(pred != y)/len(y)) # proporção de acertos

	# mostra resultados
	clf._plot(X, y, true_w = w_target)	

