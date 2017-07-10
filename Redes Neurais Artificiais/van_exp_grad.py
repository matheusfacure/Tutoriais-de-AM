import numpy as np # para computação numérica menos intensiva
import os # para criar pastas
import tensorflow as tf # para redes neurais

# criamos uma pasta para colocar os dados
if not os.path.exists('tmp'):
	os.makedirs('tmp')

# baixa os dados na pasta criada
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=True) # carrega os dados já formatados

# definindo constantes
lr = 0.01 # taxa de aprendizado
n_iter = 100 # número de iterações de treino
batch_size = 128 # qtd de imagens no mini-lote (para GDE)
n_inputs = 28 * 28 # número de variáveis (pixeis)
n_l1 = 128 # número de neurônios da primeira camada
n_l2 = 128 # número de neurônios da segunda camada
n_l3 = 128 # número de neurônios da terceira camada
n_l4 = 128 # número de neurônios da quarta camada
n_outputs = 10 # número classes (dígitos)

graph = tf.Graph() # cria um grafo
with graph.as_default(): # abre o grafo para colocar operações e variáveis

	# Camadas de Inputs
	with tf.name_scope('Inputs'):
		x_input = tf.placeholder(tf.float32, [None, n_inputs], name='X_input')
		y_input = tf.placeholder(tf.float32, [None, n_outputs], name='y_input')

	# Camada 1
	with tf.name_scope('Layer_1'):
		W1 = tf.Variable(tf.random_normal([n_inputs, n_l1]), name='Weight_1')
		b1 = tf.Variable(tf.zeros([n_l1]), name='bias_1')
		l1 = tf.nn.sigmoid(tf.matmul(x_input, W1) + b1, name='Sigmoid')
	
	# Camada 2
	with tf.name_scope('Layer_2'):
		W2 = tf.Variable(tf.random_normal([n_l1, n_l2]), name='Weight_2')
		b2 = tf.Variable(tf.zeros([n_l2]), name='bias_2')
		l2 = tf.nn.sigmoid(tf.matmul(l1, W2) + b2, name='Sigmoid')
		
	# Camada 3
	with tf.name_scope('Layer_3'):
		W3 = tf.Variable(tf.random_normal([n_l2, n_l3]), name='Weight_3')
		b3 = tf.Variable(tf.zeros([n_l3]), name='bias_3')
		l3 = tf.nn.sigmoid(tf.matmul(l2, W3) + b3, name='Sigmoid')
	
	# Camada 4
	with tf.name_scope('Laeyer_4'):
		W4 = tf.Variable(tf.random_normal([n_l3, n_l4]), name='Weight_4')
		b4 = tf.Variable(tf.zeros([n_l4]), name='bias_4')
		l4 = tf.nn.sigmoid(tf.matmul(l3, W4) + b4, name='Sigmoid')

	# camada de saída
	with tf.name_scope('Output'):
		Wo = tf.Variable(tf.random_normal([n_l4, n_outputs]), name='W_out')
		bo = tf.Variable(tf.zeros([n_outputs]), name='bias_o')
		logits = tf.add(tf.matmul(l4, Wo), bo)
		y_hat = tf.nn.softmax(logits) # converte scorer em probabilidades
	
	# função objetivo
	with tf.name_scope('Train'):
		error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits, name='error'))
		optimizer = tf.train.GradientDescentOptimizer(lr).minimize(error)

	with tf.name_scope('Gradients'):
		de_dW1, de_dW2, de_dW3, de_dW4  = tf.gradients(error, [W1, W2, W3, W4])
		tf.summary.histogram('Grads1', de_dW1)
		tf.summary.histogram('Grads2', de_dW2)
		tf.summary.histogram('Grads3', de_dW3)
		tf.summary.histogram('Grads4', de_dW4)
	
	# inicializador
	init = tf.global_variables_initializer()

	# para o tensorboard	
	summaries = tf.summary.merge_all() # funde todos os summaries em uma operação
	file_writer = tf.summary.FileWriter('logs', tf.get_default_graph()) # para escrever arquivos summaries


# abrimos a sessão tf
with tf.Session(graph=graph) as sess:
	init.run() # iniciamos as variáveis
	
	# loop de treinamento
	for step in range(n_iter+1):

		# cria os mini-lotes
		x_batch, y_batch = data.train.next_batch(batch_size)

		# cria um feed_dict
		feed_dict = {x_input: x_batch, y_input: y_batch}

		# executa uma iteração de treino
		summaries_str, _ = sess.run([summaries, optimizer], feed_dict=feed_dict)
			
		# a cada 10 iterações, salva os registros dos summaries
		if step % 10 == 0:
			file_writer.add_summary(summaries_str, step)
			d1, d2, d3, d4 = sess.run([de_dW1, de_dW2, de_dW3, de_dW4], feed_dict=feed_dict)
			print(d1.mean(), d2.mean(), d3.mean(), d4.mean())
			
file_writer.close() # fechamos o nó de escrever no disco.