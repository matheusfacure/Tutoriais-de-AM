from sklearn.datasets import load_boston # para carregar os dados
from sklearn.linear_model import LinearRegression # importa o modelo
from sklearn.model_selection import train_test_split

# carrega os dados
house_data = load_boston()
X = house_data['data']
y = house_data['target']

# separa em set de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = LinearRegression()
regr.fit(X_train, y_train)

r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)
