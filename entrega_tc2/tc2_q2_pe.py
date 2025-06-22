import numpy as np
from otimo import BFGS, PenalidadeExterior, GradienteConjugado, QuasiNewton, SecaoAurea, LagrangeanoAumentado 
import matplotlib.pyplot as plt
import time

def funcao_objetivo(x):
    P1, P2, P3 = x[0], x[1], x[2]
    return (
        0.15*P1**2 + 38*P1 + 756 +
        0.1*P2**2 + 46*P2 + 451 +
        0.25*P3**2 + 40*P3 + 1049
    )

def PL(P1, P2, P3):
    B = np.array([
        [0.000049, 0.000014, 0.000015],
        [0.000014, 0.000045, 0.000016],
        [0.000015, 0.000016, 0.000039]
    ])
    P = np.array([[P1], [P2], [P3]])
    return (P.T @ B @ P).item()

# Restrições
def g1(x): return   150 - x[0] # 150 - P1 < 0
def g2(x): return   x[0] - 600 # P1 - 600 < 0
def g3(x): return   100 - x[1] # 100 - P2 < 0
def g4(x): return   x[1] - 400 # P2 - 400 < 0
def g5(x): return   50 - x[2] # 50 - P3 < 0
def g6(x): return   x[2] - 200 # P3 - 200 < 0
def h1(x): return   x[0] + x[1] + x[2] - 850 - PL(x[0], x[1], x[2]) # P1 + P2 + P3 - 850 - PL = 0

restricoes = [g1, g2, g3, g4, g5, g6, h1]
tipo = np.array(['<', '<', '<', '<', '<', '<', '='])
x0 = np.array([-100, 50.0, 100.0])

busca_1d = SecaoAurea(precisao=1e-6)
#irrestrito = BFGS(busca_1d)
irrestrito = GradienteConjugado(unidimensional=busca_1d)

restrito = PenalidadeExterior(precisao=1e-6)

start_time = time.time()

solucao = restrito.resolva(
    funcao_objetivo, x0, restricoes, tipo,
    irrestrito, penalidade=1.0, aceleracao=1.3,
    disp=True
)

print(time.time() - start_time)


print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")
print(PL(solucao.x[0], solucao.x[1], solucao.x[2]))


print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")

plt.plot(solucao.fxhist, marker='o', color='r')

plt.xlabel('Iteração')
plt.ylabel('Função Objetivo')
plt.title('Convergência do método - Gradiente Conjugado + Penalidade Exterior')

plt.show()