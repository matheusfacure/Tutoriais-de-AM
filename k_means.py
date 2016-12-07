import pandas as pd
import numpy as np
from sklearn import model_selection, cluster
import matplotlib.pyplot as plt
from matplotlib import style
from time import time
style.use('ggplot')


def test_cluster(clf, data):

	t0 = time() 
	clf.fit(data)
	centroids = clf.cluster_centers_
	print("Tempo:", round(time()-t0, 3), "s")

	colors = 10*["g","r","c","b","k"]
	
	for color, feature in zip(clf.labels_, data.values):
		plt.scatter(feature[0], feature[1], color=colors[color], s=35)
	
	for c in range(len(clf.cluster_centers_)):
		plt.scatter(centroids[c][0], centroids[c][1],
					marker="o", color="k", s=50, linewidths=5)
	
	labels = data.columns
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])


	plt.show()


class k_means(object):

	def __init__(self, k, tol = 0.0001, max_iter = 500):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter


	def fit(self, data):

		self.labels_ = np.zeros(shape=(len(data)))

		# seleciona os centros aleatoriamente para começar
		rand_k = [np.random.randint(0, len(data)) for rand in range(self.k)]
		self.cluster_centers_ = data.ix[rand_k, :].values


		for _ in range(self.max_iter):

			# cria classes vazias para serem povoadas
			temp_class = {}
			for i in range(self.k):
				temp_class[i] = []

			# acha que ponto pertence a que centro
			for j, i in enumerate(data.values):

				# acha a cistância entre a observação i e cada centro
				dist = [np.linalg.norm(i-j) ** 2 for j in self.cluster_centers_]
					
				# classifica a observação i a um centro
				clas = dist.index(min(dist))
				temp_class[clas].append(i)
				self.labels_[j] = clas

			# cira centro antigo para verificar otimização.
			# passa por cópia e ñ por referência
			prev_centers = np.array(self.cluster_centers_)

			# atualiza os centros
			for i, _ in enumerate(self.cluster_centers_):
				self.cluster_centers_[i] = np.array(temp_class[i]).mean(axis=0)

			# verifica convergência
			var = np.sum((self.cluster_centers_ - prev_centers) / 
						prev_centers*100.0)
			
			if var < self.tol:
				break

		# converta as classes para ints
		self.labels_ = self.labels_.astype(int)
	




if __name__ == '__main__':

	#lê os dados
	#dados podem ser encontrados no link abaixo
	# http://www.cengage.com/aise/economics/wooldridge_3e_datasets/
	# OBS: dados utilizados aqui já foram convertidos de excel para csv
	data = pd.read_csv('hprice.csv', sep=',', usecols = ['price', 'sqrft'])
	data.fillna(-99999, inplace = True)

	
	# Testa regressor criado manualmente
	print('\nResultados do criado manualmente')
	clf = k_means(k=3)
	clf.fit(data)
	print(clf.cluster_centers_)
	test_cluster(clf, data)


	# Compara o regressor com o do sklearn
	print('\nComparando com os resultados do Sklearn')
	clf = cluster.KMeans(n_clusters=3)
	clf.fit(data)
	print(clf.cluster_centers_)
	test_cluster(clf, data)
