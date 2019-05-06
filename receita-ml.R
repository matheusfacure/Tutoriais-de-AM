# 1 - Carregando os Dados
# -----------------------
# Os dados estão disponíveis para download neste link:
# http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29

# Os dados são automaticamente baixados na minha pasta Downloads.
# O comando setwd navega o R até esta pasta. 
# (No Windows o caminho pode ser diferente)
setwd("~/Downloads/")

# Nossos dados são um arquivo de texto onde as entradas são 
# separadas por vírgulas. Por isso, colocaremos sep=",". Além
# Disso, as variáveis não estão nomeadas e a primeira linha
# do arquivo já contém amostra de dados. Por isso, colocaremos
# header=F. Por fim, os valores faltantes (missing) estão codificados
# como uma string "?".
dados  <- read.table("breast-cancer-wisconsin.data", sep=",", header=F, na.strings = c("?"))


# Como os nossos dados não estão originalmente com nomes nas colunas,
# nós vamos colocar esses nomes. Os nomes das variáveis pode ser encontrado
# na documentação dos dados, no link acima.
# Abaixo, criamos um vetor de strings com os nomes das variáveis
nomes <- c("Sample_code_number",
           "Clump_Thickness",   
           "Uniformity_of_Cell_Size",  
           "Uniformity_of_Cell_Shape",  
           "Marginal_Adhesion",        
           "Single_Epithelial_Cell_Size",
           "Bare_Nuclei",                 
           "Bland_Chromatin",          
           "Normal_Nucleoli",          
           "Mitoses",                    
           "Class")


features <- c("Clump_Thickness", "Uniformity_of_Cell_Size",
              "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
              "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses")

# (2 for benign, 4 for malignant)
target <- "Class"


# E agora nomeamos as colunas com a função names (não confundir com
# a variável nomes que acabamos de criar) 
names(dados) <- nomes

dados[, target] <- as.numeric(dados[, target] == 4)

# 2 - Separando os Dados
# ----------------------
# Logo após ler os dados, o próximo passo envolve separar a base de 
# dados em 3: treino, validação e teste
# Treino: Os dados de treino serão usados para treinar (ou ajustar, ou estimar)
# o modelo de aprendizado de máquina. 
# Validação: Os dados de validação serão utilizados comparar vários 
# modelos e escolher aquele com a melhor performance
# Teste: Os dados de teste serão utilizados para verificar nossa estimativa
# final de acerto.

# o seed garante consistência na aleatoriedade
set.seed(432) 

# vamos usar 20 % dos dados para teste. O código
# abaixo retira uma amostra aleatória de linhas
# de forma que essa amostra tenha 80% das linhas
# (0.8 * 32561 = 559 linhas)
id <- sample(1:nrow(dados), nrow(dados)*0.8)

# desses 80% de linhas escolhidas, nós vamos pegar 60%
# dos dados originais e chamar de linhas de treino
# (0.6 * 699 = 419 linhas)
id.treino <- sample(id, nrow(dados)*0.6)

# as linhas que não estão no teste (não estão em id)
# e nem no treino serão as linhas de validação
id.val <- id[!(id %in% id.treino)]

# agora vamos usar essas linhas para criar os 3 datasets
dados.tr <- dados[id.treino,]
dados.val <- dados[id.val,]
dados.test <- dados[-id,]

# 3 - Análise Exploratória
# ------------------------
# Análise exploratória serve para entendermos um pouco sobre os nossos dados.
# Alguns coisas importantes para se descobrir nessa análise são
# 1) A escala de cada variável (media, mínimo, máximo e quantis)
# 2) Quais variáveis são categóricas e quais são numéricas
# 3) Há valores faltantes (NAs) em alguma variável
# Essa análise deve ser sempre feita no treino.

summary(dados.tr)

# Com a função summary, podemos responder todas as perguntas acima.
# 1) Todas as nossas variáveis variam de 1 a 10.
# 2) Não temos variáveis categóricas, apenas contínuas.
# 3) A variável Bare_Nuclei contem valores faltantes (NAs)


# 4 - Pre-Processamento
# ----------------------
# Antes de passar nossos dados por um algoritmo de aprendizado de máquina
# é preciso fazer alguns pré-processamentos. Alguns algoritmos funcionam 
# melhor quando os dados estão todos centrados e escalonados
# (com média 0 e desvio padrão 1). Além disso, precisamos tratar as variáveis
# Faltantes de alguma forma. Aqui, vamos simplesmente imputar todos os dados
# faltantes com a mediana.

# install.packages("caret")
library(caret)

# Esses passos de pré-processamento envolvem uma estimação e toda estimação deve ser
# feita no dataset de treino. A função preProcess treina um pre-processador que, nesse caso,
# vai ser utilizado para normalizar as variáveis e imputar valores faltantes com a mediana. 
preProcValues <- preProcess(dados.tr[, features], method = c("center", "scale", "medianImpute"))

# Uma vez que esse pre-processador está treinado, nós podemos aplicá-lo nos dados de 
# treino, validação e teste. Fazemos isso com a função predict e passando tanto o
# pre-processador quanto os novos dados.
dados.tr = predict(preProcValues, newdata = dados.tr)
dados.val = predict(preProcValues, newdata = dados.val)
dados.test = predict(preProcValues, newdata = dados.test)

summary(dados.tr)

# 5 - Treinando um Modelo de ML
# -----------------------------
# Com os dados pre-processados, estamos prontos para treinar nosso algoritmo de 
# aprendizado de máquina. Nesse caso, vamos utilizar uma floresta aleatória. 
# A floresta aleatória treina várias árvores de decisão em amostras aleatórias dos
# dados e depois combina a previsão de todas essas árvores em uma previsão final.

# install.packages("randomForest")
library(randomForest)

# Nosso modelo vai ser ajustado para um problema de classificação, isto é, 
# prever uma variável binária (ou categórica). No nosso caso, estamos tentando
# prever se um tumor é maligno (Class = 1) ou não (Class = 0). Para isso, vamos
# Usar varias variáveis sobre o tumor, como Clump_Thickness e Uniformity_of_Cell_Size.
# Nós representamos o que queremos prever (variável target) e o que vamos usar para prever
# (features) com uma fórmula. A variável target fica a esquerda do "~". Além disso, como
# Estamos lidando com um problema de classificação, vamos dizer que a variável target é
# categórica com "as.factor(...)". As variáveis preditivas (features) vão do lado direito
# do "~" e são separadas por um "+". O "+" aqui NÃO significa que vamos somar as variáveis
# Ele quer dizer que vamos INCLUIR essas variáveis no modelo.
formula <- paste0("as.factor(", target, ")~", paste0(features, collapse="+"))

# Finalmente, vamos treinar nossa floresta aleatória. O primeiro argumento para o 
# algoritmo é a fórmula que criamos acima. Em segundo, vem os dados onde vamos treinar
# esse modelo. Nós sempre treinamos nossos modelos nos dados de treino! Os dados de validação e
# teste são apenas utilizados para verificar a qualidade do modelo, isto é, como o modelo
# se sai prevendo novos dados, diferentes daqueles que ele já viu durante o treinamento.

# Os próximos argumentos são os híper-parâmetros. Híper-parâmetros ajustam a complexidade do nosso modelo. 
# Se um modelo for muito complexo ele conseguirá ajustar perfeitamente os dados de treino, 
# isto é, suas previsões nos dados de treino serão perfeitas e ele não errará nada nessa base de dados.
# No entanto, é comum que um modelo que seja perfeito na base de treino sofra com overfitting, isto é, 
# o modelo ajusta muito bem a base de treino mas não generaliza a boa performance para novos dados, dados
# que não foram utilizados para treinar o modelo.
# Por outro lado, se um modelo for muito simples a sua performance será ruim na base de treino, pois ele
# não consegue achar os padrões nos dados. Assim, quando o modelo for fazer previsões numa nova base de dados
# também não conseguirá explorar os padrões e terá uma performance ruim.
# O grande desafio então é encontrar um modelo nem muito complexo e nem muito simples.

# Para começar, vamos treinar um modelo relativamente complexo. Será um modelo com apenas 5 árvores (ntree=5), 
# mas cada árvore poderá crescer até que cada amostra seja classificada corretamente. Isso é feito com
# maxnodes=NULL, que diz que não há limite para o crescimento de cada árvore. Por fim, o hiper-parâmetro mntry=9
# diz que cada árvore pode usar todas as nove variáveis para ajustar suas previsões.

set.seed(432) 
model <- randomForest(as.formula(formula),
                      data=dados.tr,
                      ntree=5,
                      mntry=9,
                      maxnodes=NULL)


# 6 - Fazendo previsões
# ---------------------
# Agora que temos um modelo treinado, vamos fazer previsões com eles. Passamos o modelo e os dados para
# a função predict, que faz as previsões.

pred.train = predict(model, newdata=dados.tr)
pred.val = predict(model, newdata=dados.val)

head(pred.train)

# 7 - Vendo a performance
# -----------------------
# Com as nossas previsões em mãos, vamos ver a taxa de acerto, ou acurácia do nosso modelo. A acuráveis é
# Simplesmente a quantidade de previsões corretas dividida pelo tamanho da base de dados. Aqui, para conseguir
# Esse número vamos simplesmente tirar a média dos acertos (previsto == observado)
mean(pred.train == dados.tr[,target])
mean(pred.val == dados.val[,target])

# Como era de se esperar com um modelo complexo, a nossa performance na base de treino é quase perfeita. 
# Temos uma acurácia de 0.997, o que quer dizer que praticamente todas as nossas previsões estão corretas.
# Por outro lado, nossas previsões em dados de validação, que não foram utilizados para treinar o modelo, é bem
# mais baixa: 0.957. Isso quer dizer que estamos errando mais do que 4% das nossas previsões.
# Nós não vamos observar ainda a performance de teste pois testaremos novos modelos. A base de teste só
# Pode ser olhada quando tivermos selecionado nosso modelo final.

# 8 - Ajustando a Complexidade
# ----------------------------
# Nosso primeiro modelo provavelmente está sofrendo com overfitting, isto é, boa performance de treino mas
# não tão boa performance em novos dados. Vamos tentar corrigir isso ajustando a complexidade do nosso modelo.
# Para fazer isso, vamos treinar vários modelos com complexidade (ou hiper-parâmetros) diferentes.
# Seleção de modelos é ainda um problema aberto. Existem várias formas de selecionar modelos sendo estudadas,
# mas uma que funciona bem na prática é definir um espaço de hiper-parâmetros e ir tentando combinações
# aleatórias desses hiper-parâmetros (Random Search).

# Primeiro definimos todos os possíveis hiper-parâmetros. Por exemplo, para o hiper-parâmetros mtry testaremos
# 2,3,4,5,8,10,11 e assim por diante
mtry <- c(2,3,4,5,8,9)
ntree <- c(50, 100, 150)
nodesize <- c(1, 2, 5, 10)
maxnodes <- c(2, 3, 5, 6, 10)

# Em seguida definimos o tanto de modelos que testaremos. Vamos testar 20 modelos.
n.try = 20

# Depois criamos um dataframe de 20 linhas em que cada linha é uma combinação dos hiper-parâmetros.
# Definidos acima e cada coluna é corresponde a um hiper-parâmetro. Para isso, a função sample vai
# retirar 20 amostras (com substituição) do espaço de hiper-parâmetro que definimos acima. Fazemos
# Isso para cada hiper-parâmetro.

try.df <- data.frame(mtry     = sample(mtry, n.try, replace=T),
                     ntree    = sample(ntree, n.try, replace=T), 
                     nodesize = sample(nodesize, n.try, replace=T),
                     maxnodes = sample(maxnodes, n.try, replace=T))

# Por último, criamos uma coluna vazia nesse dataframe. Essa coluna será preenchida
# com a performance de cada modelo que testaremos. Nós olharemos apenas a performance
# em dados diferentes dos usados para treinar, isto é, nos dados de validação.
# Aliás, esse é justamente o papel dos dados de validação: servir de base de comparação
# entre modelos.
try.df$performance.val <- NA

head(try.df)

set.seed(432) 
for (linha in 1:nrow(try.df)) {
  
  # treinamos um modelo com os hiper-parâmetros da linha 
  # da iteração atual. Note como estamos treinando nos dados
  # de treino: dados.tr
  model.iter <- randomForest(as.formula(formula),
                             data=dados.tr,
                             ntree=try.df[linha, "ntree"],
                             mntry=try.df[linha, "mtry"],
                             nodesize=try.df[linha, "nodesize"],
                             maxnodes=try.df[linha, "maxnodes"])
  
  # fazemos previsões nos dados de validação: dados.val
  pred.val.iter <- predict(model.iter, newdata=dados.val)
  
  # computamos a acurácia para as nossas previsões nos 
  # dados de validação
  acc.val.iter <- mean(pred.val.iter == dados.val[,target])
  
  # salvamos esse resultado na coluna de performance
  try.df[linha, "performance.val"] <- acc.val.iter
}

head(try.df)

# Agora que testamos um monte de modelos, temos que escolher o que achamos melhor.
# Um critério bem simples é pegar aquele que teve menos erros na base de validação. 
# Pode haver mais de um. Para achar esses modelos vamos ver aqueles que tiveram acurácia
# igual a acurácia máxima encontrada.
best.models <- which(try.df$performance.val == max(try.df$performance.val))

try.df[best.models, ]

# Temos 4 modelos empatados segundo esse critério. Vamos simplesmente pegar o primeiro.

# 9 - Modelo Final
# ----------------
# Depois de termos testado vários modelo precisamos criar o nosso modelo final, aquele que
# De fato usaremos para fazer novas previsões.

# Já temos nosso modelo selecionado, então não precisamos mais reservar os nossos dados
# de validação. Vamos então criar uma base final de treino que contém os dados de treino e
# validação
final.train <- rbind(dados.tr, dados.val)

# E vamos treinar um modelo com a mesma complexidade (mesmos hiper-parâmetros) do melhor
# modelo, selecionado no passo anterior.
set.seed(432) 
final.model <- randomForest(as.formula(formula),
                            data=final.train,
                            ntree=150,
                            mntry=4,
                            nodesize=2,
                            maxnodes=5)


# 10 - Performance Final
# ---------------------
# Depois de termos testados todos esses modelos e escolhido o que
# tem a melhor performance de previsão, podemos finalmente ver a nossa performance 
# nos dados de teste.

pred.test <- predict(final.model, newdata=dados.test)
mean(pred.test == dados.test[,target])

# A nossa performance final foi pior do que a performance de validação. Isso acontece. 
# Pode ser que a nossa seleção de hiper-parâmetros tenha levado a um overfit também nos
# dados de validação. De qualquer forma, esse é nossa estimativa final de performance e é 
# a que devemos reportar como a esperada.

# 11 - Prevendo Uma Nova Amostra
# ------------------------------
# Temos nosso modelo treinado. Vamos ver como utilizá-lo na prática. Imagine que um novo paciente
# chegou e traz consigo as seguintes informações, obtidas por exames médicos

new.sample <- data.frame(Clump_Thickness = 8,
                         Uniformity_of_Cell_Size = NA,
                         Uniformity_of_Cell_Shape = 1.0,
                         Marginal_Adhesion = 8.0,
                         Single_Epithelial_Cell_Size = 2.0,
                         Bare_Nuclei = 10.0,
                         Bland_Chromatin = 3.0,
                         Normal_Nucleoli = NA,
                         Mitoses = 1.0) 

# Esse novo paciente não tem os dados dos exames de Normal_Nucleoli nem de Uniformity_of_Cell_Size, por
# Isso vamos colocar como um valor faltante. 

# Antes de prever qual a probabilidade deste paciente ter um
# tumor maligno, precisamos passar os dados dele pelo pre-processador. Isso lidará com os valores
# faltantes de maneira correta.
new.sample.process <- predict(preProcValues, newdata = new.sample)
new.sample.process

# Finalmente, podemos fazer nossas previsões para esse novo paciente.
predict(final.model, newdata = new.sample.process)
predict(final.model, newdata = new.sample.process, type = "prob")

# Más notícias: Nosso modelo prevê que o tumor deste paciente é um tumor maligno (Class=1). 
# Mas isso não é tudo. Nosso modelo diz que a probabilidade deste paciente ter um tumor maligno
# é de 0.68. Talvez seja uma boa ideia o paciente fazer uns testes a mais para termos mais certeza.



