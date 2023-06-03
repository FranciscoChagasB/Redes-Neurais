# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:39:09 2023

@author: Francisco
"""

import math
import numpy as np

# Definindo os dados de entrada e saída da porta NAND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, 1, 0])

# Definindo o número de neurônios na camada escondida
n_hidden = 2

# Normalizando os dados de entrada
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

# Definindo os centróides das funções de base radial
kmeans = np.random.choice(X_norm.shape[0], n_hidden, replace=False)
centroids = X_norm[kmeans]

# Calculando as distâncias dos dados de entrada aos centróides
distances = np.zeros((X_norm.shape[0], n_hidden))
for i in range(n_hidden):
    for j in range(X_norm.shape[0]):
        distances[j][i] = math.exp(-1 * np.sum((X_norm[j] - centroids[i])**2))

# Adicionando o bias aos dados de entrada
distances = np.hstack((distances, np.ones((distances.shape[0], 1))))

# Inicializando os pesos da camada de saída
W_out = np.random.rand(n_hidden+1)

# Definindo a função de ativação da camada de saída
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Treinando a rede neural
n_epochs = 1000
learning_rate = 0.1
for epoch in range(n_epochs):
    print(f"Iteração: {epoch}")
    for i in range(X_norm.shape[0]):
        # Propagação para a camada escondida
        hidden_output = distances[i].dot(W_out)
        hidden_output = sigmoid(hidden_output)
        # Propagação para a camada de saída e cálculo do erro
        output = sigmoid(hidden_output)
        error = y[i] - output
        # Atualização dos pesos da camada de saída
        W_out += learning_rate * error * hidden_output * (1 - hidden_output) * distances[i]
        print(f"Pesos: {W_out}, Saída: {output}")

# Testando a rede neural com novos dados
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X_test_norm = (X_test - mean) / std
distances_test = np.zeros((X_test_norm.shape[0], n_hidden))
for i in range(n_hidden):
    for j in range(X_test_norm.shape[0]):
        distances_test[j][i] = math.exp(-1 * np.sum((X_test_norm[j] - centroids[i])**2))
distances_test = np.hstack((distances_test, np.ones((distances_test.shape[0], 1))))
y_pred = []
for i in range(X_test_norm.shape[0]):
    hidden_output = distances_test[i].dot(W_out)
    hidden_output = sigmoid(hidden_output)
    output = sigmoid(hidden_output)
    y_pred.append(round(output))

