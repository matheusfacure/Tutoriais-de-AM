import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


class perceptron(object):
	
	def __init__(self, w = None, lrate = 0.5, maxiter=5, tol = 0, plot_=False):
		self.w = w
		self.lrate = lrate
		self.maxiter = maxiter
		self.tol = tol
		self.plot_ = plot_


	def __update_w(self, neg_examples, pos_examples):

		for xi in neg_examples:
			activation = np.dot(xi, self.w)
			if activation >= 0:
				self.w = self.w - (xi.T * self.lrate)

		for xi in pos_examples:
			activation = np.dot(xi, self.w)
			if activation < 0:
				self.w = self.w + (xi.T * self.lrate)

	

	def __eval_perceptron(self, neg_examples, pos_examples):

		error = 0
		for xi in neg_examples:
			activation = np.dot(xi, self.w)
			if activation >= 0:
				error += 1

		
		for xi in pos_examples:
			activation = np.dot(xi, self.w)

			if activation < 0:
				error += 1

		return error



	def fit(self, X_train, y_train):
		X_train = np.array(X_train)
		X_train = np.insert(X_train, 0, 1, 1) # adiciona vies
		y_train = np.array(y_train)

		# inicia os pessos aleatóriamente
		if self.w is None:
			self.w = np.random.normal(3, 1, len(X_train[0]))

		# separa os exemplos positivos dos negativos
		pos = X_train[y_train == 1, :]
		neg = X_train[y_train == 0, :]

		count = 0
		while True:

			error = self.__eval_perceptron(neg, pos)
			
			if self.plot_:

				print(error)
				colors = ['red','green']
				plt.scatter(X_train[:,1], X_train[:,2], c=y,
					cmap=matplotlib.colors.ListedColormap(colors))
				w = self.w
				plt.plot([-10,30], [ (-w[0]+30*w[1])/w[2],
								   (-w[0]-30*w[1])/w[2] ],'k')
				plt.show()


			self.__update_w(neg, pos)


			count += 1
			if error < self.tol or count > self.maxiter:
				break

			


if __name__ == '__main__':
	pos = np.random.normal(10, 3, 10)
	neg = np.random.normal(20, 3, 10)
	
	x1 = np.array(range(len(pos) + len(neg)))
	x2 = np.append(pos, neg)
	X = np.array([x1, x2]).T
	y = np.append(np.array([1] * len(pos)), np.array([0] * len(pos)))

	clf = perceptron(plot_ = True)
	clf.fit(X, y)

