from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Carregando o conjunto de dados de câncer de mama
data = load_breast_cancer()
X = data.data
y = data.target

# Dividindo os dados em conjuntos de treinamento e teste usando validação cruzada hold-out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Pré-processamento dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Redimensionando os dados de entrada
X_train = X_train.reshape(-1, X.shape[1], 1, 1)
X_test = X_test.reshape(-1, X.shape[1], 1, 1)

# Criação do modelo de CNN
model = Sequential()
model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X.shape[1], 1, 1)))
model.add(MaxPooling2D((2, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=2)
