import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np # para computação numética menos intensiva
import os # para criar pastas
from matplotlib import pyplot as plt # para mostrar imagens
import tensorflow as tf # para redes neurais
plt.style.use('ggplot')

# criamos uma pasta para salvar o modelo
if not os.path.exists('tmp'): # se a pasta não existir
	os.makedirs('tmp') # cria a pasta para guardar os dados

# baixa os dados na pasta criada e carrega os dados
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)


def fully_conected_layer(inputs, n_neurons, activation=tf.nn.sigmoid):
	'''
	Adiciona os nós de uma camada ao grafo TensorFlow e
	retorna o tensor de saída da camada.
	Args:
		inputs: um tensor de entrada da camda
		n_neurons: a qtd de neurônios da camada
		activation: a função de ativação da camada (padrão: tf.nn.sigmoid)

	'''
	# define as variáveis da camada
	n_inputs = int(inputs.get_shape()[1]) # pega o formato dos inputs
	# usa uma semente para garantir a consitência na inicialização aleatória
	W = tf.Variable(tf.truncated_normal([n_inputs, n_neurons], seed=1)) 
	b = tf.Variable(tf.zeros([n_neurons]), name='biases')
	
	# operação linar da camada
	layer = tf.add(tf.matmul(inputs, W), b, name='linear_transformation')
	
	# aplica não linearidade, se for o caso
	if activation is None:
		return layer
	else:
		return activation(layer)
	

def	leaky_relu(z, leak=0.01):
	'''Cria uma função de ativação leaky ReLU'''
	return tf.maximum(leak * z, z)


def net(X_tensor, y_tensor, activation=tf.nn.sigmoid):
	'''
	Adiciona ao grafo os nós de uma rede neural.
	Retorna um tuple (opt, acc), com o nó de otimização da rede neural e o nó de acurácia
	'''

	# Monta uma rede neural simples, com duas camadas e 512 neurônios por camada
	l1 = fully_conected_layer(X_tensor, n_neurons=512, activation=activation)
	l2 = fully_conected_layer(l1, n_neurons=512, activation=activation)
	logit = fully_conected_layer(l2, n_neurons=10, activation=None)

	# computa o erro e faz o nó de otimização
	error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logit))
	train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

	# calcula acurácia
	correct = tf.nn.in_top_k(logit, y_tensor, 1) # calcula obs corretas
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # converte para float32

	return (train_step, accuracy)


if __name__ == '__main__':
	
	graph = tf.Graph()
	with graph.as_default():

		x_input = tf.placeholder(tf.float32, [None, 28*28])
		y_input = tf.placeholder(tf.int64, [None])

		# cria uma rede neural para cada função de ativação
		sig_step, sig_acc = net(x_input, y_input, activation=tf.nn.sigmoid)
		tanh_step, tanh_acc = net(x_input, y_input, activation=tf.nn.tanh)
		relu_step, relu_acc = net(x_input, y_input, activation=tf.nn.relu)
		elu_step, elu_acc = net(x_input, y_input, activation=tf.nn.elu)
		leaky_relu_step, leaky_relu_acc = net(x_input, y_input, activation=leaky_relu)

		# junta todos os passos te otimização e acurácias. Vamos iterar por eles depois.
		opt_steps = [sig_step, tanh_step, relu_step, elu_step, leaky_relu_step]
		acuracies = [sig_acc, tanh_acc, relu_acc, elu_acc, leaky_relu_acc]

		init = tf.global_variables_initializer()


	with tf.Session(graph=graph) as sess:
		init.run() # iniciamos as variáveis

		# pega 1000 amostras do set de teste para avaialção
		test_x, text_y = data.test.next_batch(1000)
		test_dict = {x_input: test_x, y_input: text_y}

		# loop de treinamento
		all_accs = []
		for step in range(1001):

			# monta os mini-lotes
			x_batch, y_batch = data.train.next_batch(64)
			feed_dict = {x_input: x_batch, y_input: y_batch}

			# uma iteração de treino para cada rede
			for opt in opt_steps:
				sess.run(opt, feed_dict=feed_dict) # roda uma iteração de treino
			
			# a cada 10 passos, calcula a acurácia no set de teste.
			if step % 10 == 0:
				acc_list = []
				for acc in acuracies:
					a = sess.run(acc, feed_dict=test_dict)
					acc_list.append(a)

				all_accs.append(acc_list)

	df = pd.DataFrame(np.array(all_accs) * 100)
	df.plot()
	plt.show()
