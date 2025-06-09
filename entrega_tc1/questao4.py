
import numpy as np
from scipy.optimize import minimize

def funcaoobjetivo(x):
    x1, x2 = x
    return 0.6382*x1**2 + 0.3191*x2**2 - 0.2809*x1*x2 - 67.906*x1 - 14.29*x2

def gradiente(x):
    x1, x2 = x
    df_dx1 = 2 * 0.6382 * x1 - 0.2809 * x2 - 67.906
    df_dx2 = 2 * 0.3191 * x2 - 0.2809 * x1 - 14.29
    return np.array([df_dx1, df_dx2])

x1 = 40
x2 = -77
x = np.array([x1, x2])

#hessiana da funcao objetivo
hessiana = np.array([[2*0.6382, -0.2809],
                     [-0.2809, 2*0.3191]])

hessiana_inv = np.linalg.inv(hessiana)

otimo  = x - hessiana_inv@gradiente(x)

# resultado

print(otimo)






