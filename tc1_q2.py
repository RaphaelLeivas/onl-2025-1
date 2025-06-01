# vamos usar gradiente conjuggado, pois a funcao é continua, diferenciavel e tem muitas variaveis

import numpy as np
from scipy.optimize import minimize
from arquivos_tc1.otimo import GradienteConjugado, SecaoAurea

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função-objetivo
def funcaoobjetivo(w):
    
    # Carregar os dados do problema
    dados = np.load('arquivos_tc1/questao2_dados.npz')
    X = dados['X'] # Entrada: matriz de dimensões m x n
    y = dados['y'] # Saída: rótulos binários
    m, n = X.shape
    lambd = 0.1

    h = sigmoid(X @ w)
    loss = -np.mean(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
    reg = (lambd / 2) * np.sum(w**2)
    return loss + reg

otimizacao_unidimensional = SecaoAurea(precisao=1e-2, passo=1e-3, maxaval=200)
gc = GradienteConjugado(otimizacao_unidimensional)

# Gera solucao inicial
n = 500
w0 = np.zeros(n)
resultado = gc.resolva(funcaoobjetivo, w0)

# Pesos otimizados
# custo_final = ???
# numero_iteracoes = ???
# numero_avaliacoes = ???


# print(f"Custo final: {custo_final:.4f}")
# print(f"Número de iterações: {numero_iteracoes}")
# print(f"Número de avaliações da função-objetivo: {numero_avaliacoes}")