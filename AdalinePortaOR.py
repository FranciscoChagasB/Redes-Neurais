# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:01:20 2023

@author: Francisco
"""

import numpy as np

# A classe Adaline representa uma Rede Neural Adaline. 
class Adaline:
    def __init__(self, size, eta=0.3, epochs=100):
        self.size = size
        self.eta = eta # passo de aprendizado.
        self.epochs = epochs # número máximo de iterações que podem ser realizadas.
        self.w = np.zeros(self.size + 1) # inicializa os pesos com zeros.
        self.errors = []

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.w[1]) + self.w[0] # calcula a soma ponderada.
        activation_function = 1 if weighted_sum >= 0 else 0 # aplica a função de ativação.
        return activation_function

    def train(self, trainingset, labels):
        
        for epoch in range(self.epochs):
            epoch_error = 0
            for inputs, label in zip(trainingset , labels):
                prediction = self.predict(inputs)
                error = label - prediction
                epoch_error += error ** 2
                self.w[0] += self.eta * error # atualiza o bias
                self.w[1] += self.eta * error * inputs # atualiza os pesos
            self.errors.append(epoch_error)
            if epoch_error == 0:
                print(f"Convergiu após {epoch + 1} iterações")
                break
        print(f"Iterações: {epoch + 1}")
        print(f"Erro final: {self.errors[-1]}")
        print(f"Pesos: {self.w}")

# Treinamento para porta OR
trainingset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

adaline = Adaline(size=1)
adaline.train(trainingset, labels)