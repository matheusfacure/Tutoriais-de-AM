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

	# plotando os resultados
	plt.figure()
	ax1 = plt.subplot2grid((1,1), (0,0))
	data.plot.scatter(x='sqrft', y='price', alpha=1, s=45,
		label=None, ax = ax1)
	plt.scatter(centroids[:, 1], centroids[:, 0],
            marker='*', s=100, linewidths=3,
            color='k', zorder=10)
	plt.show()


class k_means(object):

	def __init__(self, k, tol = 0.0001, max_iter = 10):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter


	def fit(self, data):

		# seleciona os centros aleatoriamente para começar
		rand_k = [np.random.randint(0, len(data)) for rand in range(self.k)]
		self.cluster_centers_ = data.ix[rand_k, :].values


		for _ in range(self.max_iter):

			# cria classes vazias para serem povoadas
			classes = {}
			for i in range(self.k):
				classes[i] = []

			# acha que ponto pertence a que centro
			for i in data.values:

				# acha a cistância entre a observação i e cada centro
				dist = [np.linalg.norm(i-j) for j in self.cluster_centers_]
					
				# classifica a observação i a um centro
				clas = dist.index(min(dist))
				classes[clas].append(i)

			# cira centro antigo para verificar otimização.
			# passa por cópia e ñ por referência
			prev_centers = np.array(self.cluster_centers_)

			# atualiza os centros
			for i, _ in enumerate(self.cluster_centers_):
				self.cluster_centers_[i] = np.array(classes[i]).mean(axis=0)

			# verifica convergência
			var = np.sum((self.cluster_centers_ - prev_centers) / 
						prev_centers*100.0)
			
			if var < self.tol:
				break



		







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


	# # Compara o regressor com o do sklearn
	# print('\nComparando com os resultados do Sklearn')
	# clf = cluster.KMeans(n_clusters=3)
	# clf.fit(data)
	# print(clf.cluster_centers_)
	# test_cluster(clf, data)
