import pandas as pd
import numpy as np
from sklearn import model_selection, neighbors
from matplotlib import style
from time import time
from scipy import stats

 

class knn_clf(object):

	def __init__(self, k = 3):
		self.k = k

	
	def fit(self, X_train, y_train):
		X = np.insert(X_train, 0, y_train, 1)
		self.lookup_table = X


	def predict(self, X_test):

		answers = np.zeros(X_test[:, 0].shape)
		
		# para cada observação no set de teste
		for a_i, row in enumerate(X_test):

			# achamos as distâncias para cada observação no set de treino
			dist = np.linalg.norm(self.lookup_table[:, 1:] - row, axis=1)
			
			# acoplamos uma coluna de distância na tabela de busca
			lookup_temp = np.insert(self.lookup_table, 0, dist, 1)

			# ordena segundo a distância calculada
			lookup_temp = lookup_temp[lookup_temp[:, 0].argsort()]

			# acha os votos das k observações mais perto
			votes = lookup_temp[:self.k, 1]
				
			# acha o mais votado e adiciona à resposta
			answers[a_i] = stats.mode(votes)[0]

		return answers


	def score(self, X_test, y_test):

		y_perd = self.predict(X_test)
		
		return np.sum(y_perd == y_test) / len(y_test)



if __name__ == '__main__':
	
	# lê os dados
	# Dados podem ser encontrados no link abaixo
	# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
	# OBS: As colunas foram nomeadas nos dados usados aqui
	var_list = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
	 'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei',
	 'bland_chromation', ' normal_nucleoli', 'mitoses', 'class']

	data = pd.read_csv('breast-cancer-wisconsin.data.txt', 
		na_values = '?', usecols = var_list)

	data.fillna(-99999, inplace = True)
	X = np.array(data.drop(['class'], 1))
	y = np.array(data['class'])
	
	# separa em treino e teste
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
																test_size=0.2)

	# treina e testa algoritmo criado
	print('Aplicando o classificador criado manualmente')
	t0 = time()
	clf = knn_clf(k=5)
	clf.fit(X_train, y_train)
	print('Pontuação:', clf.score(X_test, y_test))
	print("Tempo do criado manualmente:", round(time()-t0, 3), "s")


	# treina e testa algoritmo do sklearn
	print('\nAplicando o classificador do sklearn')
	clf_test = neighbors.KNeighborsClassifier(n_neighbors = 5)
	clf_test.fit(X_train, y_train)
	print('Pontuação:', clf_test.score(X_test, y_test))
	print("Tempo do classificador do sklearn:", round(time()-t0, 3), "s")

