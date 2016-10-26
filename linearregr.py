import pandas as pd
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from matplotlib import style
from time import time
style.use('ggplot')


class linear_regr(object):

	def __init__(self):
		pass

	def fit(self, X, y):
		# adiciona coluna de 1 para achar intercepto
		X = np.insert(X, 0, 1, 1)

		# estima os betas
		# Fórmula por álgebra linear
		betas = np.dot( np.dot( np.linalg.inv(np.dot(X.T, X)), X.T), y)
		
		self.betas = betas
		self.coef = self.betas[1:]
		self.intercept = self.betas[0]


	def score(self, X, y):
		X = np.insert(X, 0, 1, 1)
		self.y_pred = np.dot(X, self.betas)
		self.sqr_err = (y - self.y_pred) ** 2
		
		# R^2
		self.score = ( np.sum( (self.y_pred - np.mean(y)) ** 2)  /
						np.sum( (y - np.mean(y)) ** 2) ) 
		return(self.score)



if __name__ == '__main__':

	# gera os dados
	X, y, coef = datasets.make_regression(n_samples=10000000, n_features=3,
									  n_informative=2, noise=20,
									  coef=True, random_state=0)

	print('Aplicando o regressor criado manualmente')
	t0 = time()
	regr = linear_regr()
	regr.fit(X, y)
	print('Coeficientes:', regr.coef)
	print('Intercepto:', regr.intercept)
	print('R^2:', regr.score(X, y))
	print("Tempo do criado manualmente:", round(time()-t0, 3), "s")

	print('\nComparando com os resultados do Sklearn')
	# Compara o regressor com o do sklearn
	t0 = time() 
	regr_test = linear_model.LinearRegression()
	regr_test.fit(X, y)
	print('Coeficientes:', regr_test.coef_)
	print('Intercepto:', regr_test.intercept_)
	print('R^2:', regr_test.score(X, y))
	print("Tempo do sklearn:", round(time()-t0, 3), "s")
