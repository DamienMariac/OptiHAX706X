#%%
import numpy as np
from scipy.optimize import fsolve

#j'ai fais par substitution

def equation(x1):
    x2 = x1**3 - 1 
    eq = 3 * x1**2 * x2 + 3 * x1**5 
    return eq


x1_solution = fsolve(equation, 0.8)  # Un point proche de la racine cube de 1/2

x1 = x1_solution[0]
x2 = x1**3 - 1
minimum_J = x1**3 * x2

print("Solution trouvée :")
print("x1 =", x1)
print("x2 =", x2)
print("Minimum de J(x1, x2) =", minimum_J)


# %%
