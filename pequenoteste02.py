"""
                            PEQUENO TESTE 02

Este pequeno teste aborda a aplicação do método da Seção Áurea para a
minimização de uma função unidimensional. O problema de otimização abordado é
o seguinte:

    Minimizar f(x) = x^4 - 3*x^2 + x

O desafio neste teste é implementar a atualização dos pontos a, b, u e v do 
método da Seção Áurea. A atualização desses pontos é realizada nas linhas
58-59, 65-66, 76-77 e 83-84. Você só precisa completar as linhas de código com 
a atualização correta dos pontos.
"""
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})

# Definição da função-objetivo
def func(x):
    return x**4 - 3*x**2 + x

# Parâmetros gerais
a = -2 # Início do intervalo
b = 0 # Fim do intervalo
fa = func(a) # Valor da função no início do intervalo
fb = func(b) # Valor da função no fim do intervalo
precisao = 1e-3 # Precisão da busca
navaliacoes = 2 # Número de avaliações da função

""" Método da Seção Áurea """

# Calcula o comprimento do intervalo
L = b-a
    
# Determina dois pontos médios
u = b - .618*L
v = a + .618*L
    
# Avalia
fu = func(u)
fv = func(v)
navaliacoes = 2

# Registra a curva da função-objetivo e o intervalo inicial em uma figura
x = np.linspace(-2.2, 2)
fx = func(x)
plt.plot(x, fx, '--k')
plt.plot([a, b], [fa, fb], '*r', markersize=10, label='Intervalo inicial')

# Enquanto o meu intervalo não for reduzido a um tamanho suficientemente
# pequeno
while (b-a) > precisao:        
        
    if fu < fv:
        # elimina tudo à direita de v
            
        """ IMPLEMENTE AQUI A ATUALIZAÇÃO DE B E F(B) """
        b = v
        fb = func(v) 
            
        # Atualiza o novo comprimento do intervalo
        L = b-a
            
        """ IMPLEMENTE AQUI A ATUALIZAÇÃO DE V E F(V) """
        v = a+.618*L
        fv = func(v)
            
        # Calcula o novo u
        u = b -.618*L
        fu = func(u)
        
    # Se fu > fv
    else:
        # elimina tudo à esquerda de u
            
        """ IMPLEMENTE AQUI A ATUALIZAÇÃO DE A E F(A) """
        a = u
        fa = func(u)
            
        # Atualiza o novo comprimento do intervalo
        L = b-a
            
        """ IMPLEMENTE AQUI A ATUALIZAÇÃO DE U E F(U) """
        u = b -.618*L
        fu = func(u)
            
        # Calcula o novo u
        v = a + .618*L
        fv = func(v)
        
    navaliacoes += 1

print('Número de avaliações: %d' %navaliacoes)
    
# A aproximação do meu ótimo é o meio do meu intervalo
xotimo = (a+b)/2
fxotimo = func(xotimo)

# Registra o ótimo na figura
plt.plot(xotimo, fxotimo, '*y', markersize=10, label='Ótimo')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()