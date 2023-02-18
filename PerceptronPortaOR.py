# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:21:17 2023

@author: Francisco
"""

import random

entradas = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]] #Entrada para porta NAND
target = [0, 1, 1, 1]
eta = 0.7 #Passo de aprendizado
maxiterations = 20 #Numero máximo de iterações que podem ocorrer
#Definindo os pesos e o bias
w1 = random.uniform(-0.7, 0.7)
w2 = random.uniform(-0.7, 0.7)
w0 = random.uniform(-0.7, 0.7)
x0 = 1

error = random.uniform(-0.7, 0.7)
count = 0

while count < maxiterations and error != 0:
    error = 0
    for i, x in enumerate(entradas):
        #Calculando saida da rede
        output = x[0]*w1 + x[1]*w2 + w0*x0
        if output >= 0:
            output = 1
        else:
            output = 0
            
        error += abs(target[i] - output)
        
        #Alterando os pesos
        w1 += eta*(target[i] - output)*x[0] # regra delta
        w2 += eta*(target[i] - output)*x[1]
        w0 += eta*(target[i] - output)
        
        print("Saída " + str(output) + " target " + str(target))
        print("Erro " + str(error))
    count += 1
print("Iterações: " + str(count))
print("Erro final: " + str(error))
print("w1: " + str(w1) + ", w2: " + str(w2) + ", w0: " + str(w0))