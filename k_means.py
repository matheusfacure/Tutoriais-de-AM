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










if __name__ == '__main__':

	#lê os dados
	#dados podem ser encontrados no link abaixo
	# http://www.cengage.com/aise/economics/wooldridge_3e_datasets/
	# OBS: dados utilizados aqui já foram convertidos de excel para csv
	data = pd.read_csv('hprice.csv', sep=',', usecols = ['price', 'sqrft'])
	data.fillna(-99999, inplace = True)

	print('\nResultados do criado manualmente')
	# Compara o regressor com o do sklearn
	clf = cluster.KMeans(n_clusters=3)
	test_cluster(clf, data)


	print('\nComparando com os resultados do Sklearn')
	# Compara o regressor com o do sklearn
	clf = cluster.KMeans(n_clusters=3)
	test_cluster(clf, data)