import numpy as np

#Entradas da porta NOR
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#Saídas da porta NOR
saidas = np.array([[1], [0], [0], [0]])

#Número de neurônios na camada oculta
neuronios_oculta = 4

#Número de neurônios na camada de saída
neuronios_saida = 4

#Pesos iniciais aleatórios para a camada oculta
pesos_oculta = np.random.rand(2, neuronios_oculta)

#Pesos iniciais aleatórios para a camada de saída
pesos_saida = np.random.rand(neuronios_oculta, neuronios_saida)

#Função de ativação sigmoid
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#Função para a propagação para frente (forward propagation)
def forward(X, pesos_oculta, pesos_saida):
    camada_oculta = sigmoid(np.dot(X, pesos_oculta))
    camada_saida = sigmoid(np.dot(camada_oculta, pesos_saida))
    return camada_saida

#Taxa de aprendizado
learning_rate = 0.7

# Definir o número de épocas de treinamento
num_epochs = 5000

# Realizar o treinamento da rede
for epoch in range(num_epochs):
    # Propagação para frente (forward propagation)
    camada_oculta = sigmoid(np.dot(entradas, pesos_oculta))
    camada_saida = sigmoid(np.dot(camada_oculta, pesos_saida))

    # Calcular o erro na camada de saída
    erro_saida = saidas.T - camada_saida
    media_absoluta_erro = np.mean(np.abs(erro_saida))
    
    # Calcular a derivada da função de ativação sigmoid na camada de saída
    derivada_saida = erro_saida * (camada_saida * (1 - camada_saida))
    
    # Calcular a correção dos pesos da camada de saída
    correcao_saida = np.dot(camada_oculta.T, derivada_saida)

    # Calcular o erro na camada oculta
    erro_oculta = np.dot(derivada_saida, pesos_saida.T) * (camada_oculta * (1 - camada_oculta))

    # Calcular a correção dos pesos da camada oculta
    correcao_oculta = np.dot(entradas.T, erro_oculta)

    # Atualizar os pesos da camada de saída
    pesos_saida += learning_rate * correcao_saida

    # Atualizar os pesos da camada oculta
    pesos_oculta += learning_rate * correcao_oculta

    # Imprimir a média absoluta do erro a cada 1000 épocas
    if epoch % 1000 == 0:
        print("Época:", epoch)
        print("Erro médio absoluto:", media_absoluta_erro)
        print("Saída atual:", camada_saida.T)
        print("\n")