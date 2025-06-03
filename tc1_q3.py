import numpy as np
from arquivos_tc1.otimo import HookeJeeves

# Função de custo negativa (para maximização do lucro)
def funcaoobjetivo(x):
    d, t, m = x  # Desconto, tempo, orçamento
    VB = 100000  # Vendas básicas
    CB = 10000  # Custo fixo inicial
    
    # Receita
    f1 = -0.005 * d**2 + 0.2 * d
    f2 = 0.05 * t
    receita = VB * (1 + f1 + f2) * np.log(1 + m)

    # penalidade
    penalidade = 0
    PENALIDADE_REST = 1e6
    PENALIDADE_D = 5000
    PENALIDADE_T = 2000

    if d > 30:
        penalidade = penalidade + PENALIDADE_D

    if t > 15:
        penalidade = penalidade + PENALIDADE_T


    if d < 0 or d > 50 or t < 1 or t > 30 or m < 1000 or m > 50000:
        penalidade = penalidade + PENALIDADE_REST
    
    # Custo
    custo_total = CB + m + penalidade

    # Lucro
    lucro = receita - custo_total
    
    # Lembre-se que eu quero maximizar o lucro e meu algoritmo de otimização minimiza a função objetivo
    return lucro * (-1)

# Chute inicial
ponto_inicial = [10, 10, 10000]

""" IMPLEMENTE AQUI A CHAMADA DO ALGORTIMO DE OTIMIZAÇÃO """
metodo = HookeJeeves()
resultado = metodo.resolva(funcaoobjetivo, ponto_inicial)

print(resultado)

# Resultados
d = resultado.x[0]
t = resultado.x[1]
m = resultado.x[2]
lucro = funcaoobjetivo(resultado.x) * (-1)

print(f"Parâmetros ótimos: Desconto = {d:.2f}%, Tempo = {t:.2f} dias, Orçamento = ${m:.2f}")
print(f"Lucro máximo estimado: R${lucro:.2f}")