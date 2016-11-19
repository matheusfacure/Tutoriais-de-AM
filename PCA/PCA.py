import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as skPCA
import matplotlib.pyplot as plt
from matplotlib import style
from time import time
style.use('ggplot')
np.random.seed(12)

class PCA(object):

	def __init__(self, n_components):
		
		self.n_components = n_components

	
	def fit(self, X):
		S = np.dot(X.T, X)
		auto_val, auto_vect = np.linalg.eig(S)
		
		sort_vect = np.argsort(auto_val)[::-1][:self.n_components]
		
		self.auto_val = auto_val[sort_vect]
		self.components_ = auto_vect[:, sort_vect]
		self.m = X.mean(axis=0)


	def transform(self, X):
		print(np.round(self.auto_val, 3))
		print(np.round(self.components_, 3))		
		return np.dot((X - self.m), self.components_)



def plot_pca(dados, tranf_dados, components):
	
	pca1 = components[0]
	pca2 = components[1]

	for ii, jj in zip(tranf_dados, dados):
		plt.scatter(pca1[0] * ii[0], pca1[1] * ii[0], color = 'red')
		plt.scatter(pca2[0] * ii[1], pca2[1] * ii[1], color = 'c')
		plt.scatter(jj[0], jj[1])
	plt.xlim([-4, 4])
	plt.ylim([-4, 4])	
	plt.show()



if __name__ == '__main__':
		
	x1 = np.random.normal(1, 0.5, 50)
	x2 = 2 * x1 + np.random.normal(0, 0.5, len(x1))
	x3 = x1
	dados = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3}).values

	pca = skPCA(n_components=2)
	pca.fit(dados)
	tranf_dados = pca.transform(dados)
	plot_pca(dados, tranf_dados, pca.components_)

	mypca = PCA(n_components=2)
	mypca.fit(dados)	
	tranf_dados = mypca.transform(dados)
	plot_pca(dados, tranf_dados, pca.components_)
