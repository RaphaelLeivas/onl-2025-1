import numpy as np
from scipy.optimize import minimize

# Objective function
def objective(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

# Scipy defines it as: constraint >= 0 for ineq
# So we write: 2 - (x0 + x1) >= 0
cons = (
    {
        'type': 'ineq',
        'fun': lambda x: 1 - (x[0] + 2 * x[1])
    },
    {
        'type': 'ineq',
        'fun': lambda x: 1 - (x[0] ** 2 + x[1])
    },
    {
        'type': 'ineq',
        'fun': lambda x: 1 - (x[0] ** 2 - x[1])
    },
    {
        'type': 'eq',
        'fun': lambda x: 2 * x[0] + x[1] - 1
    },
)

# Bounds for x0 and x1
bnds = ((0, 1), (-0.5, 2))

# Initial guess
x0 = [0.5, 0.5]

# Perform optimization
result = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

# Show result
print("Optimal solution:", result.x)
print("Objective value:", result.fun)