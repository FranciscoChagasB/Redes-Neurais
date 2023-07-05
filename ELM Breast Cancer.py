from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Carregando o conjunto de dados de câncer de mama
data = load_breast_cancer()
X = data.data
y = data.target

# Normalização "zscore" nos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em conjuntos de treinamento e teste usando validação "hold-out"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementação da Rede de Aprendizado Extremo (ELM)
class ELM:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Inicialização dos pesos aleatórios para a camada de entrada para a camada oculta
        self.W = np.random.uniform(-1, 1, (self.n_input, self.n_hidden))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        H = self.sigmoid(np.dot(X, self.W))

        # Cálculo dos pesos aleatórios para a camada oculta para a camada de saída
        self.beta = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self.sigmoid(np.dot(X, self.W))
        y_pred = np.dot(H, self.beta)
        return np.round(y_pred).astype(int)

    def compute_loss(self, X, y):
        H = self.sigmoid(np.dot(X, self.W))
        y_pred = np.dot(H, self.beta)
        return np.mean((y - y_pred) ** 2)

# Criando e treinando a ELM
n_input = X_train.shape[1]
n_hidden = 100  # Número de neurônios na camada oculta
n_output = 2  # Número de classes
elm = ELM(n_input, n_hidden, n_output)
elm.train(X_train, y_train)

# Realizando as previsões
y_pred = elm.predict(X_test)

# Calculando a precisão e o loss
accuracy = accuracy_score(y_test, y_pred)
loss = elm.compute_loss(X_test, y_test)
print("Accuracy:", accuracy)
print("Loss:", loss)
