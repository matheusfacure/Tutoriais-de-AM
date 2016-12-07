
# Regressão Linear
<br/>
<br/>

## Conteúdo

- [Pré-requisitos](#pre_requisitos)
- [Justificativa matemática](#justificativa_matematica)
- [Desenhando e testando o algoritmo](#desenhando_e_testando_o_algoritmo)
- [Indo um pouco além: Projeção com Erros Ortogonais](#indo_um_pouco_alem)
- [Introduzindo relações não lineares](#introduzindo_relacoes_nao_lineares)
- [Recomendações e Considerações Finais](#recomendacoes_e_consideracoes_finais)
- [Ligações Externas](#ligacoes_externas)

<br/>
<a id='pre_requisitos'></a>
## Pré-requisitos

É preciso ter um conhecimento básico de Python, incluindo o mínimo de Python orientado à objetos. Caso não saiba programar, os cursos de [Introdução à Ciencia da Computação](https://br.udacity.com/course/intro-to-computer-science--cs101/) e [Fundamentos de Programação com Python](https://br.udacity.com/course/programming-foundations-with-python--ud036/) fornecem uma base suficiente sobre programação em Python e Python orientado à objetos, respectivamente. Além disso, é necessário ter conhecimento das bibliotecas de manipulação de dados Pandas e Numpy. Alguns bons tutoriais são o [Mini-curso 1](https://br.udacity.com/course/machine-learning-for-trading--ud501/) do curso de Aprendizado de Máquina para Negociação, o site [pythonprogramming.net](https://pythonprogramming.net/data-analysis-python-pandas-tutorial-introduction/) ou o primeiro curso do DataCamp em [Python](https://www.datacamp.com/getting-started?step=2&track=python).

Para entender o desenvolvimento do algoritmo de regressão linear é preciso ter o conhecimento de introdução à álgebra linear. Na UnB, a primeira parte do curso de Economia Quantitativa 1 já cobre o conteúdo necessário. Caso queira relembrar ou aprender esse conteúdo, o curso online do MIT de [Introdução à Álgebra Linear](https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8) fornece uma boa base sobre a matemática que será desenvolvida nos algoritmos de aprendizado de máquina.

Conhecimento de cálculo e principalmente otimização é fundamental para o entendimento dos algoritmos de aprendizado de máquina, que muitas vezes são encarados explicitamente como problemas de otimização. Uma noção de cálculo multivariado também ajudará na compreensão dos algoritmos, visto que muitas vezes otimizaremos em várias direções.



<br/>
<a id='justificativa_matematica'></a>
## Justificativa matemática

Imagine que temos dados em tabelas, sendo que cada linha é uma observação e cada coluna uma variável. Nos então escolhemos uma das colunas para ser nossa variável dependente y (aquela que queremos prever) e as outras serão as variáveis independentes (X). Nosso objetivo é aprender como chegar das variáveis independentes na variável dependente, ou, em outras palavras, prever y a partir de X. Note, que X é uma matriz nxd, em que n é o número de observações e d o número de dimensões; y é um vetor coluna nx1. Podemos definir o problema como um sistema de equações, em que cada equação é uma observação:

$\begin{cases} 
w_0 + w_1 x_1 + ... + w_d x_1 = y_1 \\
w_0 + w_1 x_2 + ... + w_d x_2 = y_2 \\
... \\
w_0 + w_1 x_n + ... + w_d x_n = y_n \\
\end{cases}$

Normalmente, $n > d$, isto é, temos mais observações que dimensões. Sistemas assim costumam não ter solução; há muitas equações e poucas variáveis para ajustar. Intuitivamente, pese que, na prática, muitas coisas afetam a variável y. Principalmente se ela for algo de interesse das ciências humanas, como, por exemplo, preço, desemprego, felicidade... E muitas das coisas que afetam y não podem ser coletadas como dados; as equações acima não tem solução porque não temos todos os fatores que afetam y. 

Para lidar com esse problema, vamos adicionar nas equações um termo erro $\varepsilon$ que representará os fatores que não conseguimos observar, erros de medição, etc.

$$\begin{cases} 
w_0 + w_1 x_{11} + ... + w_d x_{1d} +  \varepsilon_1 = y_1 \\
w_0 + w_1 x_{21} + ... + w_d x_{2d} + \varepsilon_2 = y_2 \\
... \\
w_0 + w_1 x_{n1} + ... + w_d x_{nd} + \varepsilon_3 = y_n \\
\end{cases}$$

Em forma de matriz:

$$ \begin{bmatrix}
    1 & x_{11} & ... & x_{1d} \\
    1 & x_{21} & ... & x_{2d} \\
    \vdots &  \vdots&  \vdots &  \vdots \\
    1 & x_{n1} & ... & x_{nd} \\
\end{bmatrix}
\times
\begin{bmatrix}
    w_0 \\
    w_1 \\
    \vdots \\
    w_d \\
\end{bmatrix}
+
\begin{bmatrix}
    \varepsilon_0 \\
    \varepsilon_1 \\
    \vdots \\
    \varepsilon_n \\
\end{bmatrix}
=
\begin{bmatrix}
    y_0 \\
    y_1 \\
    \vdots \\
    y_n \\
\end{bmatrix}$$


$$X_{nd} \pmb{w}_{d1} + \pmb{\epsilon}_{n1} = \pmb{y}_{n1}$$



Para estimar a equação acima, usaremos a técnica de Mínimos Quadrados Ordinários (MQO): queremos achar os $\pmb{\hat{w}}$ que minimizam os $n$ $ \varepsilon^2 $, ou, na forma de vetor, $\pmb{\epsilon}^T \pmb{\epsilon}$. Por que minimizar os erros quadrados? Bom, não há uma resposta certa para isso. Note que os erros variam para mais e para menos e tem média zero, de forma que a soma deles será sempre muito próxima de zero. Então temos que fazer algo para que todos os erros sejam positivos. Poderíamos minimizar os erros absolutos, mas a o quadrado dos erros também funciona e deixa a matemática bem mais simples: 

\begin{equation}
\begin{split}
    \pmb{\epsilon}^T  \pmb{\epsilon} &= (\pmb{y} - \pmb{\hat{w}}X)^T(\pmb{y} - \pmb{\hat{w}} X) \\
             &= \pmb{y}^T \pmb{y} - \pmb{\hat{w}}^T X^T \pmb{y} - \pmb{y}^T X \pmb{\hat{w}} + \pmb{\hat{w}} X^T X \pmb{\hat{w}} \\
             &= \pmb{y}^T \pmb{y} - 2\pmb{\hat{w}}^T X^T \pmb{y} + \pmb{\hat{w}} X^T X \pmb{\hat{w}}
\end{split}
\end{equation}

Aqui, usamos o fato que que $\pmb{\hat{w}}^T X^T \pmb{y}$ e $\pmb{y}^T X \pmb{\hat{w}}$ são simplesmente escalares $1x1$ e a transposta de um escalar é o mesmo escalar: $\pmb{\hat{w}}^T X^T \pmb{y} = (\pmb{\hat{w}}^T X^T \pmb{y})^T = \pmb{y}^T X \pmb{\hat{w}}$. Derivando em $\pmb{\hat{w}}$ e achando a CPO:


$$\frac{\partial \pmb{\epsilon}^T \pmb{\epsilon}}{\partial \pmb{\hat{w}}} = -2X^T\pmb{y} + 2X^T X \pmb{\hat{w}} = 0$$


Derivando mais uma vez para checar a CSO chegamos em $2X^TX$, que é positiva definida se as colunas de X forem independentes. Temos então um ponto de mínimo quando:


$$ \pmb{\hat{w}} = (X^T X)^{-1} X^T \pmb{y}$$


Bom, parece que chegamos em algo interessante. Nos nossos dados temos $X$ e $\pmb{y}$, então podemos achar $\hat{\pmb{w}}$ facilmente: basta substituir os valores na fórmula! O próximo passo e desenhar o algoritmo e ver como ele se sai em dados reais.

OBS:  
1) Para mais detalhes, veja [este](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf) passo a passo da Universidade de Stanford.  
2) Seria possível chegar em uma fórmula para os vários $\hat{w_i}$ apenas com cálculo multivariado, sem usar álgebra linear. Embora a forma com álgebra linear seja mais difícil (pelo menos foi para mim) ela vai nos ajudar no entendimento de como o algoritmo funciona. Álgebra linear é uma ferramente poderosa de abstração e a vasta maioria dos algoritmos de aprendizado de máquina usam álgebra linear em suas derivações, então é bom já irmos nos acostumando. 

<br/>
<a id='desenhando_e_testando_o_algoritmo'></a>
## Desenhando e testando o algoritmo


```python
import pandas as pd # para ler os dados em tabela
import numpy as np # para álgebra linear
from sklearn import linear_model, model_selection, datasets # para comparar o nosso algoritmo com o de mercado
import matplotlib.pyplot as plt # para fazer gráfico
from matplotlib import style
from time import time # para ver quanto tempo demora
style.use('ggplot')
np.random.seed(1)

class linear_regr(object):

    def __init__(self):
        pass


    def fit(self, X_train, y_train):
        # adiciona coluna de 1 nos dados
        X = np.insert(X_train, 0, 1, 1)

        # estima os w_hat
        w_hat = np.dot( np.dot( np.linalg.inv(np.dot(X.T, X)), X.T), y_train)
                                    # (X^T * X)^-1 * X^T * y
        self.w_hat = w_hat
        self.coef = self.w_hat[1:]
        self.intercept = self.w_hat[0]


    def predict(self, X_test):
        X = np.insert(X_test, 0, 1, 1) # adiciona coluna de 1 nos dados
        y_pred = np.dot(X, self.w_hat) # X * w_hat = y_hat
        return y_pred

```

Ok, teoria justificada e algoritmo pronto. Vamos ver se ele consegue aprender os $\hat{\pmb{w}}$ de dados reais.
OBS: Os dados podem ser encontrados em http://www.cengage.com/aise/economics/wooldridge_3e_datasets/.  
  
  
Lendo e processando os dados:


```python
data = pd.read_csv('../data/hprice.csv', sep=',').ix[:, :6] # lendo os dados
data.fillna(-99999, inplace = True) # preenchendo valores vazios
X = np.array(data.drop(['price'], 1)) # escolhendo as variável independentes
y = np.array(data['price']) # escolhendo a variável dependente

# separa em treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)
data.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>assess</th>
      <th>bdrms</th>
      <th>lotsize</th>
      <th>sqrft</th>
      <th>colonial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300.0</td>
      <td>349.1</td>
      <td>4</td>
      <td>6126</td>
      <td>2438</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>370.0</td>
      <td>351.5</td>
      <td>3</td>
      <td>9903</td>
      <td>2076</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>191.0</td>
      <td>217.7</td>
      <td>3</td>
      <td>5200</td>
      <td>1374</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>195.0</td>
      <td>231.8</td>
      <td>3</td>
      <td>4600</td>
      <td>1448</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>373.0</td>
      <td>319.1</td>
      <td>4</td>
      <td>6095</td>
      <td>2514</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Treinando, testando e comparando o regressor.


```python
t0 = time()
regr = linear_regr()
regr.fit(X_train, y_train)
print("Tempo do criado manualmente:", round(time()-t0, 3), "s")

# medindo os erros
y_hat = regr.predict(X_test) # prevendo os preços

print('Média do erro absoluto: ', np.absolute((y_hat - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat - y_test) / y_test)).mean())

# comparando com o de mercado
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("\n\nTempo do de mercado:", round(time()-t0, 3), "s")

# medindo os erros
y_hat = regr.predict(X_test) # prevendo os preços
w_hat = regr.intercept_
w_hat = np.append(w_hat, regr.coef_)

print('Média do erro absoluto: ', np.absolute((y_hat - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat - y_test) / y_test)).mean())

```

    Tempo do criado manualmente: 0.097 s
    Média do erro absoluto:  34.9234990043
    Média do erro relativo:  0.122915533711
    
    
    Tempo do de mercado: 0.254 s
    Média do erro absoluto:  34.9234990043
    Média do erro relativo:  0.122915533711


Nada mal... O erro previsto é, na média, apenas 12,2% diferente do preço real/observado. Note que o algoritmo aprendeu os parâmetros $\hat{\pmb{w}}$ com uma parte dos dados e usou para prever dados que nunca tinha visto, mostando uma boa capacidade de generalização.

O nosso algoritmo produz os mesmos resultados do de mercado, então podemos saber que não erramos nada. Além disso, o nosso algoritmo é mais rápido que o de mercado, mas essa diferênça é insignificante, em termos práticos. Cabe aqui uma observação: **não reinvente a roda!**. Na prática, se existe um bom algorítmo já feito, use-o! Não é preciso fazer o algorítmo do zero sempre, basta importar o do [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)! Aqui, estamos recriando os algorítmos apenas para melhor entendimento de como ele funciona, mas não com intenção de usar nossa criação na prática. Além disso, os algorítmos já pronto são muuuito melhores e mais rápidos que o nosso. O modelo de regressão linear é apenas uma exceção devido à sua simplicidade


A vantágem do modelo de regressão linear é o que chamamos de um modelo caixa branca: nos sabemos exatamente como ele aprende os parâmetros e ainda nos oferece capacidade interpretativa por meios deles. Infelizmente, a capacidade interpretativa depende de um aprofundamento que não é a intenção desse tutorial. Caso queira se aprofundar no alorítmo, veja [1](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf) ou [2](https://www.coursera.org/learn/erasmus-econometrics).

Outra vantágem da regressão linear por MQO é que o processo de treinamento é muuuuuito rápido treinar. Muito mesmo. Mesmo com milhões de dados, é possível estimar os parâmetros em menos de um segundo. Além disso, uma vez treinado, o regressor ocupa muito pouco espaço, pois só armazena o vetor $\pmb{\hat{w}}$.

Vale uma nota de atenção: esse algoritmo é a base da econometria e da ciência de dados inferencial no geral. Aqui só podemos abordá-lo brevemente. Ainda há problemas de iferência (saber se os coeficientes são estatisticamente significantes), de interpretação em outras escalar, de hipóteses assumidas e o que fazer quando elas são violadas. Tenha isso em mente na hora de usá-lo! Muita coisa ficou incompleta aqui. Caso tenha interesse em se aprofundar no assunto, apontaremos fontes externas para isso no final do tutorial.

<br/>
<a id='indo_um_pouco_alem'></a>   
## Indo um pouco além: Projeção com Erros Ortogonais

Nós chegamos em um fórmula muito útil $\pmb{\hat{w}} = (X^T X)^{-1} X^T \pmb{y}$, mas, para mim, ainda não está claro o que essa fórmula faz, além de minimizar os erros quadrados. O objetivo aqui é entender melhor como o algorítmo funciona por meio de vizualização e exemplos. 

Bom, a primeira coisa que notamos é que $X\pmb{\hat{w}}$ produz $\pmb{\hat{y}}$ e não $\pmb{y}$. Há uma diferênça entre $\pmb{\hat{y}}$ e $\pmb{y}$ que é um resíduo $\pmb{\epsilon}$. Podemos então definir:

\begin{equation}
\begin{split}
    \pmb{\epsilon} &= \pmb{y} - X \pmb{\hat{w}} \\
             &= \pmb{y} - X (X^T X)^{-1} X^T \pmb{y} \\
             &= [I -  X (X^T X)^{-1} X^T] \pmb{y} \\
             &= M\pmb{y}
\end{split}
\end{equation}

Além disso:

\begin{equation}
\begin{split}
    \pmb{\hat{y}} &= \pmb{y} - \pmb{e} \\
             &= [I - M]\pmb{y} \\
             &= X (X^T X)^{-1} X^T \pmb{y}
\end{split}
\end{equation}

Chamaremos $X (X^T X)^{-1} X^T$ de $P$. A matriz $P$ transforma $\pmb{y}$ em $\pmb{\hat{y}}$, mas como? De alguma forma, eu acredito que entender essa matriz é a chave para vizualizar como o algorítmo de MQO funciona. Vamos criar um exemplo hipotético com poucos números para facilitar a vizualização.


```python
X = np.array([[1, 2],
              [2, 1],
              [3, 4],
              [5, 1],
              [2, 6],
              [3, 3]])
w = np.array([[4], 
              [1]])

y =  np.dot(X, w) + np.reshape(np.random.normal(0, 0.5, 6), (6, 1))

np.round(y, 2)
```




    array([[  6.81],
           [  8.69],
           [ 15.74],
           [ 20.46],
           [ 14.43],
           [ 13.85]])




```python
P = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
y_hat = np.dot(P, y)
y_hat
```




    array([[  6.01718147],
           [  8.70752746],
           [ 15.8336541 ],
           [ 20.10540092],
           [ 14.25225327],
           [ 14.72470894]])



Note como $\pmb{\hat{y}}$ representa a relação linear entre $X$ e $y$ retirando parte do ruido. Por exemplo, para a primeira observação, não fosse o ruido, $y_1$ seria $6$ e $\hat{y_1}$ é bem mais próximo de $6$ do que $y_1$. Se todas as variáveis relevantes estão em $X$, a matrix $P$ pode então ser vista como um filtro de ruido. A forma como eu gosto de vizualizar $P$ é como uma matriz projeção: $P$ projeta o vetor $\pmb{y}$ em $\pmb{\hat{y}}$ de forma que o reíduo seja ortogonal a $\pmb{\hat{y}}$. O resíduo ser ortogonal significa que ele não tem nenhuma relação com $\pmb{\hat{y}}$ e se ele também não tiver correlação com $\pmb{y}$, então estaremos achando a melhor relação entre as variável dependentes e independentes.

Mas isso chama a atenção para um problema muito sério: normalmente não temos todas as variáveis relevantes para explicar um fenômeno. O que acontece nesse caso?

Vamos refazer o exemplo, mas agora, suponha que não possâmos medir a segunda variável.


```python
x = np.delete(X, 1, 1)
x
```




    array([[1],
           [2],
           [3],
           [5],
           [2],
           [3]])



Nesse caso, $x$ é apenas um vetor e $X (X^T X)^{-1} X^T$ se reduz para $\frac{1}{k} X X^T$ em que $k = X^T X$. 


```python
P = np.dot(x, x.T) / np.dot(x.T, x)
y_hat = np.dot(P, y)
y_hat

```




    array([[  4.69497763],
           [  9.38995526],
           [ 14.08493288],
           [ 23.47488814],
           [  9.38995526],
           [ 14.08493288]])



Note que obtemos agora um resultado divergente do que seria o real sem o ruído. Para a primeira observação, por exemplo, sem ruído teriamos 6, mas a nossa estimativa é menor do que 4! Note que a variável que mais contribui para $y$ é $x_1$ (contribui 4x mais), e mesmo que a tenhamos, simplemente não conseguimos achar o resultado correto se nos falta $x_2$. Lembre que $P$ projeta o vetor $\pmb{y}$ em $\pmb{\hat{y}}$ de forma que os resíduos sejam ortogonais, ou seja, de forma que eles não tenham nenhuma relação com $\pmb{\hat{y}}$. Isso ainda é verdade, mas agora o resíduo tem relação com $\pmb{y}$! Isso acontece porque $\pmb{\epsilon}$ incorpora tudo o que não conseguimos observar e nesse caso não podemos observar uma das variáveis que afeta $\pmb{y}$. Por isso, assumir que $\pmb{y}$ é não depende de $\pmb{\epsilon}$ uma hipótese falha e o nosso erro de previsão sobre bastante.

No entanto, essa é ainda a melhor estimativa que podemos fazer com os dados que temos (lembre que assumimos que não podiamos observar $x_2$). Como sabemos disso? Bom, ainda estamos minimizando os erros quadrados, esse é um argumento. Do ponto de vista de projeções, nós ainda estamos projetando $y$ em $\hat{y}$ de forma que os resíduos sejam ortogonais à $\hat{y}$:



```python
e = y - y_hat
np.round(np.dot(e.T, y_hat), 10)
```




    array([[-0.]])



Viu? O produto interno entre dois vetores ortogonais é zero, então é de se esperar que tenhamos conseguido zero no produto interno acima: $\pmb{\epsilon}$ e $\pmb{y}$ são ortogonais! Como estamos em apenas duas dimenções, podemos vizualizar as projeções facilmente. Lembre-se que, do ponto de vista geométrico, ortogonalidade se reflete em vetores perpendiculares. 


```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y_hat, color='k')
plt.scatter(x, y)
for i, xi in enumerate(list(x)):
    plt.plot([xi, xi], [y[i], y_hat[i]], color='red')
plt.show()
```


![png](output_18_0.png)


Mas espere um minuto, isso não parece nada ortogonal... A linha vermelha não está paralela à preta! Eu quebrei a cabeça por horas com esse problema e ainda acho ele um tanto difícil então preste atenção e mostraremos como essas linhas são sim ortogonais.

Primeiro, lembre que a linha preta é a equação $\pmb{\hat{y}} = \pmb{x}\pmb{\hat{w}}$. A unica variável que influencia $\pmb{\hat{y}}$ é, nesse caso, $\pmb{x}$. Pos isso, fizemos o gráfico acima em apenas duas dimenções: $x$ e $y$. No entanto, o gráfico acima também tem $\pmb{y}$ (os pontos azuis). Mas $\pmb{y}$, além de depender de $\pmb{x}$ também depende de $\pmb{\epsilon}$: $\pmb{y} = \pmb{x}\pmb{\hat{w}} + \pmb{\epsilon}$. Há então uma dimenção faltando no gráfico acima se queremos vizualizar a relação entre $\pmb{\epsilon}$, $\pmb{x}$, $\pmb{\hat{y}}$ e $\pmb{y}$!

Assim, quando adicionamos a dimenção relativa à  $\pmb{\epsilon}$ conseguimos ver como os resíduos são ortogonais ao plano definido por $\pmb{\hat{y}} = \pmb{x}\pmb{\hat{w}}$. Agora sim, podemos vizualizar o que a matemática já tinha nos mostrado apenas com números!


```python
%matplotlib notebook
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

w_hat = np.dot( np.dot( np.linalg.inv(np.dot(x.T, x)), x.T), y)

fig = plt.figure()
ax = fig.gca(projection='3d')
X, E = np.meshgrid(x, e)
Y_hat = X*w_hat + 0
surf = ax.plot_surface(X, E, Y_hat, rstride=1, cstride=1, color='b')

ax.scatter(x, e, y_hat, c='k', marker='o')
ax.scatter(x, e, y, c='g', marker='o', s = 50)

plt.show()

```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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">


<br/>
<a id='introduzindo_relacoes_nao_lineares'></a>   
## Introduzindo relações não lineares

Até agora modelamos a relação entre $X$ e $y$ de forma linear. No entanto, a maioria das relações no mundo real **não** segue uma dinâmica linear! Pense no preço da casa, por exemplo. Pode ser que o aumento de preço de uma casa de 2 quartos para uma casa com 3 quartos seja diferente do aumento de preço de uma casa de 6 para 7 quartos! Uma relação linear não captura preços marginais decrescentes, por exemplo. Nos podemos facilmente introduzir não linearidade no modelo transformando as variáveis X, com uma transformação polinomial ou logarítmica, por exemplo. Vamos focar na transformação polinomial.

Até agora, estávamos modelando um polinômio de grau 1. Podemos elevar cada variável ao quadrado subir o grau do polinômio:

$$\begin{cases} 
w_0 + w_1 x_{11} + ...  + w_d x_{1d} +  w_1 x^2_{11} + ...  + w_d x^2_{1d} + \varepsilon_1 = y_1 \\
w_0 + w_1 x_{21} + ...  + w_d x_{2d} +  w_1 x^2_{21} + ...  + w_d x^2_{2d} + \varepsilon_2 = y_2 \\
... \\
w_0 + w_1 x_{n1} + ...  + w_d x_{nd} +  w_1 x^2_{n1} + ...  + w_d x^2_{nd} + \varepsilon_n = y_n \\
\end{cases}$$

Na forma de matriz, nada muda. Só devemos lembrar que $X$ nesse caso é diferente dos dados originais, porque cada variável virou duas, uma elevada a 1 e outra a 2. Aqui, vamos usar $X^*$ para explicitar essa diferença.

$$X^* \pmb{w}^* + \pmb{\epsilon} = \pmb{y}$$

Explicitando:

$$X^* \pmb{w}^* + \pmb{\epsilon} = \pmb{y}$$
$$X_{nd} \pmb{w}_{d1} + X_{nd}^*  \pmb{w}^*_{d1} +  \pmb{\epsilon} = \pmb{y}$$

Em que

$\pmb{w}^*  = \begin{bmatrix}
    x_1 w_{0+d} \\
    x_2 w_{1+d} \\
    \vdots \\
    x_d w_{d+d} \\
\end{bmatrix}$

Nosso algoritmo não muda nada. A única diferença é que os nossos inputs devem ser transformados para adicionar os componentes quadráticos.


```python
X = np.array(data.drop(['price'], 1))
y = np.array(data['price'])

# adicionando os componentes quadráticos
X_star = X ** 2
X = np.concatenate((X, X_star), axis=1)[:, :8]

# separa em treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)

regr = linear_regr()
regr.fit(X_train, y_train)

# medindo os erros
y_hat_in = regr.predict(X_train) # na amostra
y_hat_out = regr.predict(X_test) # fora da amostra

print('Na amostra')
print('Média do erro absoluto: ', np.absolute((y_hat_in - y_train)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat_in - y_train) / y_train)).mean())

print('\nFora da Amostra')
print('Média do erro absoluto: ', np.absolute((y_hat_out - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat_out - y_test) / y_test)).mean())
```

    Na amostra
    Média do erro absoluto:  25.3809238247
    Média do erro relativo:  0.0935190765106
    
    Fora da Amostra
    Média do erro absoluto:  32.7347956879
    Média do erro relativo:  0.117806543062


Podemos ver que, comparado ao modelo de apenas um grau, nossa regressão no polinômio de grau 2 tem erro absoluto médio 2 pontos menor; erro relativo também caiu 1%. 

Algumas observações:  
1) Temos nos dados uma variável binária (0,1) e quando fazemos o quadrado disso, obtemos a mesma variável. Ter uma variável repetida nos dados é um problema, pois uma uma coluna de $X$ é combinação linear da outra, fazendo com que $X^T X$ não seja inversível. Para resolver esse problema, retiramos a cópia da variável binária após elevar os dados ao quadrado.  
2) Nem sempre aumentar o grau do polinômio diminuirá os erros. Polinômios de grau muito alto terão problemas de superajustamento, com erros na amostra muito baixos, mas erros fora da amostra bem altos, indicando baixa capacidade de generalização. Veja abaixo o que ocorre quando utilizamos um polinômio de grau 6.


```python
X = np.array(data.drop(['price'], 1))
y = np.array(data['price'])

# adicionando os componentes de maior grau
X_star2 = X[:, :4] ** 2
X_star3 = X[:, :4] ** 3
X_star4 = X[:, :4] ** 4
X_star5 = X[:, :4] ** 5

X = np.concatenate((X, X_star2), axis=1)
X = np.concatenate((X, X_star3), axis=1)
X = np.concatenate((X, X_star4), axis=1)
X = np.concatenate((X, X_star5), axis=1)


# separa em treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)

regr = linear_regr()
regr.fit(X_train, y_train)

# medindo os erros
y_hat_in = regr.predict(X_train) # na amostra
y_hat_out = regr.predict(X_test) # fora da amostra

print('Na amostra')
print('Média do erro absoluto: ', np.absolute((y_hat_in - y_train)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat_in - y_train) / y_train)).mean())

print('\nFora da amostra')
print('Média do erro absoluto: ', np.absolute((y_hat_out - y_test)).mean())
print('Média do erro relativo: ', np.absolute(((y_hat_out - y_test) / y_test)).mean())
```

    Na amostra
    Média do erro absoluto:  23.3475338079
    Média do erro relativo:  0.0862915856205
    
    Fora da amostra
    Média do erro absoluto:  52.3164084616
    Média do erro relativo:  0.180609636905


<br/>
<a id='recomendacoes_e_consideracoes_finais'></a>   
## Recomendações e Considerações Finais
O modelo de regressão linear acha um hiperplano linear que minimiza a soma dos resíduos quadrados entre a variável dependente (saída) prevista pelo hiperplano e a variável dependente observada. Para estimar os parâmetros $\pmb{w}$ do hiperplano é necessário que nenhuma das variáveis independentes (entradas) seja combinação linear de outras. A não linearidade pode ser facilmente introduzida no modelo por meio de transformações diretamente nas variáveis dependentes. Essas transformações podem ser logaritmo de uma variável, quadrado de uma variável, etc.

Em problemas de aprendizado de máquina, é recomendável utilizar o modelo de regressão linear como uma primeira tentativa, devido à sua simplicidade e capacidade interpretativa. Das vantagens do modelo de regressão linear podemos destacar:

1) É de fácil interpretação. A variável de saída $y$ pode ser vista como uma soma das variáveis de entrada $X$ ponderada pelos parâmetros $w$. Assim, é possível saber diretamente qual variável de $X$ mais contribui para a variável de saída: a correspondente ao parâmetros com maior valor absoluto.  
2) É um modelo rápido de treinar, uma vez que o ponto de minimização dos erros quadrados pode ser encontrada analiticamente, sem necessidade de métodos iterativos.  
3) Uma vez treinado, o regressor ocupa puco espaço de RAM.  
4) Produz bons resultados preditivos. Normalmente, o erro do modelo linear é apenas um pouco mais alto do que os obtidos com algoritmos de aprendizado de máquina mais complexo.  

No entanto, o modelo vem com sérias desvantagens, o que nos motivará a procurar outros algoritmos mais complexos e melhores:

1) A não linearidade tem que ser incluída à mão. O modelo não aprende a forma que se dão as relações não lineares.  
2) Não é robusto à outliers. Não representa bem certas estruturas de dado, como pode ser visto no [Quarteto de Anscombe](https://pt.wikipedia.org/wiki/Quarteto_de_Anscombe), podendo produzir resultados enganosos.  
3) Se duas variáveis de $X$ são altamente correlacionadas, o modelo vai ser sensível ao ruído. Recomenda-se utilizar Análise de Componentes Principais nas variáveis altamente correlacionadas e alimentar o modelo linear apenas com o primeiro componente principal.  

<br/>
<a id='ligacoes_externas'></a>   
## Ligações Externas

1) Para colocar a mão na massa:  
O módulo do sklearn oferece boas implementações do algoritmo de regressão [linear](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) e [polinomial](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions).  

<br/>
2) Para entender mais:  
O item sobre regressão no capítulo 2 do livro [Introduction to Machine Learning](https://mitpress.mit.edu/books/introduction-machine-learning) mostra bem o dilema entre superajustamento e subajustamento. A universidade de Stanford também tem um ótimo [documento](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf) sobre regressão linear, muito mais aprofundado do que o tratado aqui.  
Se você não entendeu muito bem o modelo de regressão linear eu aconselho procurar vários tutoriais e ver cada um deles até entender. Regressão linear é a pedra fundamental sobre a qual os outros algorítmos de aprendizado de máquina serão construidos. Seja diferindo, seja melhorando o modelo de regressão linear, os algorítmos mais complexos sempre tem alguma forma de comparação com o modelo visto aqui. Segue então uma lista de videos explicando regressão linear:  
https://www.youtube.com/watch?v=D8PNnttuGZk&index=36&list=PLAwxTw4SYaPl0N6-e1GvyLp5-MUMUjOKo  
https://www.youtube.com/watch?v=udJvijJvs1M&list=PLAwxTw4SYaPkQXg8TkVdIvYv4HfLG7SiH&index=195  
https://www.youtube.com/watch?v=kJvASBvZpw0&list=PLAwxTw4SYaPnVUrK_vL3r9tP6kuwAEzgQ&index=465  


<br/>
3) Para ir além:  
O TensorFlow tem um ótimo [tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/linear/overview.html#what-is-a-linear-model) de como construir modelos lineares. Ao longo dele, faz diversas comparações com as redes neurais, que podem ser derivadas com pouca complexidade a partir dos modelos lineares.




