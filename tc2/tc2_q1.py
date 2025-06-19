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
    return P * (x2 + x1 * np.sqrt(2)) / (x1 ** 2 * np.sqrt(2) + 2 * x1 * x2) - 20  # A - 20 <= 0

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

# Configuração
restricoes = [restricao1, restricao2, restricao3, restricao4, restricao5, restricao6, restricao7]
tipos = np.array(['<', '<', '<', '>', '>', '<', '<'])
x0 = np.array([1 ,3])

busca_1d = SecaoAurea(precisao=1e-6)
irrestrito = BFGS(unidimensional=busca_1d)
# irrestrito = HookeJeeves()
restrito = LagrangeanoAumentado(precisao=1e-6)
# restrito = PenalidadeExterior(precisao=1e-6)
# restrito = PenalidadeInterior(precisao=1e-2)

# Resolução
start_time = time.time()

solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
                          irrestrito, penalidade=1.0,  aceleracao=2.0,
                          disp=True)
# solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
#                           irrestrito, penalidade=100, desaceleracao=1e-6,
#                           disp=True)


print(time.time() - start_time)

print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")

plt.plot(solucao.fxhist, marker='o', color='r')

plt.xlabel('Iteração')
plt.ylabel('Função Objetivo')
plt.title('Convergência do método - BFGS + Lagrandeano Aumentando')

plt.show()