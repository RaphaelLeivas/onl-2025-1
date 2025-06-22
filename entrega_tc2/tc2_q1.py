import numpy as np
from otimo import PenalidadeExterior, BFGS, SecaoAurea, PenalidadeInterior, LagrangeanoAumentado, NelderMeadSimplex, DFP, HookeJeeves
import matplotlib.pyplot as plt
import time

# constantes
P = 20

def objetivo(x):
    x1, x2 = x
    return 2 * np.sqrt(2) * x1 + x2

def restricao1(x):
    x1, x2 = x
    return P * (x2 + x1 * np.sqrt(2)) / ((x1 ** 2) * np.sqrt(2) + 2 * x1 * x2) - 20 # A - 20 <= 0 

def restricao2(x):
    x1, x2 = x
    return P / (x1 + x2 * np.sqrt(2)) - 20  # A - 20 <= 0

def restricao3(x):
    x1, x2 = x
    return (P * (-1) * x2) / (x1 ** 2 * np.sqrt(2) + 2 * x1 * x2) + 5 # A + 5 <= 0

def restricao4(x): 
    x1, x2 = x
    return x1 - 0.1 # x1 >= 0.1

def restricao5(x): 
    x1, x2 = x
    return x2 - 0.1 # x2 >= 0.1

def restricao6(x): 
    x1, x2 = x
    return x1 - 5 # x1 <= 5

def restricao7(x): 
    x1, x2 = x
    return x2 - 5 # x2 <= 5

def check_restricoes(rest_list, rest_tipos, sol):
    for i, rest in enumerate(rest_list):
        if rest_tipos[i] == '>':
            if rest(sol) < 0: print(f"Restricao {i + 1} nao atendida")
        if rest_tipos[i] == '<':
            if rest(sol) > 0: print(f"Restricao {i + 1} nao atendida")

# Configuração
restricoes = [restricao1, restricao2, restricao3, restricao4, restricao5, restricao6, restricao7]
tipos = np.array(['<', '<', '<', '>', '>', '<', '<'])
x0 = np.array([1, 3])

busca_1d = SecaoAurea(precisao=1e-3)
irrestrito = BFGS(unidimensional=busca_1d)
# irrestrito = NelderMeadSimplex()
restrito = LagrangeanoAumentado(precisao=1e-3)
# restrito = PenalidadeExterior(precisao=1e-3)
# restrito = PenalidadeInterior(precisao=1e-3)

# Resolução
start_time = time.time()

solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
                          irrestrito, penalidade=1,  aceleracao=2.0,
                          disp=True)
# solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
#                           irrestrito, penalidade=1e4, desaceleracao=1e-4,
#                           disp=True)

check_restricoes(restricoes, tipos, solucao.x)

print(time.time() - start_time)

print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")

plt.plot(solucao.fxhist, marker='o', color='r')

plt.xlabel('Iteração')
plt.ylabel('Função Objetivo')
plt.title('Convergência do método - BFGS + Lagrangeano Aumentado')

plt.show()