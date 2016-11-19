import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, datasets
import matplotlib.pyplot as plt
from matplotlib import style
from time import time
style.use('ggplot')


class linear_regr(object):

	def __init__(self):
		pass


	def fit(self, X_train, y_train):
		# adiciona coluna de 1 para achar intercepto
		X = np.insert(X_train, 0, 1, 1)

		# estima os betas
		# Fórmula por álgebra linear:
		# https://en.wikipedia.org/wiki/Linear_regression#Estimation_methods
		betas = np.dot( np.dot( np.linalg.inv(np.dot(X.T, X)), X.T), y_train)
		
		self.betas = betas
		self.coef = self.betas[1:]
		self.intercept = self.betas[0]


	def predict(self, X_test):
		X = np.insert(X_test, 0, 1, 1)
		y_pred = np.dot(X, self.betas)
		return y_pred


	def score(self, X_test, y_test):
		y_pred =  self.predict(X_test)
		self.sqr_err = (y_test - y_pred) ** 2
		
		# R^2
		score = ( np.sum( (y_pred - np.mean(y_test)) ** 2)  /
						np.sum( (y_test - np.mean(y_test)) ** 2) ) 
		return(score)



if __name__ == '__main__':

	# # gera os dados para teste de velocidade com muitas observações
	# # para comparar a velocidade entre o regressor criado aqui e o do sklearn
	# # remova os # do bloco abaixo e comente as linhas do bloco de leitura do
	# # arquivo csv
	# X, y, coef = datasets.make_regression(n_samples=10000000, n_features=3,
	# 								  n_informative=2, noise=20,
	# 								  coef=True, random_state=0)
	

	#lê os dados
	#dados podem ser encontrados no link abaixo
	# http://www.cengage.com/aise/economics/wooldridge_3e_datasets/
	# OBS: dados utilizados aqui já foram convertidos de excel para csv
	data = pd.read_csv('hprice.csv', sep=',')
	data.fillna(-99999, inplace = True)
	X = np.array(data.drop(['price'], 1))
	y = np.array(data['price'])

	
	# separa em treino e teste
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
																test_size=0.2)

	print('Aplicando o regressor criado manualmente')
	t0 = time()
	regr = linear_regr()
	regr.fit(X, y)
	print('R^2:', regr.score(X, y))
	print("Tempo do criado manualmente:", round(time()-t0, 3), "s")


	print('\nComparando com os resultados do Sklearn')
	# Compara o regressor com o do sklearn
	t0 = time() 
	regr_test = linear_model.LinearRegression()
	regr_test.fit(X, y)
	print('R^2:', regr_test.score(X, y))
	print("Tempo do sklearn:", round(time()-t0, 3), "s")
