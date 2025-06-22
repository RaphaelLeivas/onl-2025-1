import numpy as np
from scipy.optimize import minimize

# constantes
P = 20

# Objective function
def objective(x):
    return 2 * np.sqrt(2) * x[0] + x[1]

# Scipy defines it as: constraint >= 0 for ineq
# So we write: 2 - (x0 + x1) >= 0
cons = (
    {
        'type': 'ineq',
        'fun': lambda x: 20 - (P * (x[1] + x[0] * np.sqrt(2)) / ((x[0] ** 2) * np.sqrt(2) + 2 * x[0] * x[1]))
    },
    {
        'type': 'ineq',
        'fun': lambda x: 20 - (P / (x[0] + x[1] * np.sqrt(2)))
    },
    {
        'type': 'ineq',
        'fun': lambda x: - 5 - ((P * (-1) * x[1]) / (x[0] ** 2 * np.sqrt(2) + 2 * x[0] * x[1]))
    }
)

# Bounds for x0 and x1
bnds = ((0.1, 5), (0.1, 5))

# Initial guess
x0 = [1, 3]

# Perform optimization
result = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

# Show result
print("Optimal solution:", result.x)
print("Objective value:", result.fun)