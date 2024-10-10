# %%
from scipy.optimize import fsolve
import numpy as np

# Constantes
V = 1000
S = 10

# Système d'équations dérivées des contraintes et du Lagrangien
def system(x):
    x0, x1, x3, x4 = x  # x2 a été isolé à 100
    
    x2 = V / (x0 * x1)  # Substituer la valeur de x2 à partir de C1
    
    return [
        x1 + x2 + x3 * x2 * x1 + x4 * x1,                   # dL/dx1
        x0 + x2 + x3 * x0 * x2 + x4 * x0,                   # dL/dx2
        x0 + x1 + x3 * x0 * x1,                             # dL/dx3
        x0 * x1 - S                                         # dL/dp2 : contrainte C2=0
    ]

# Point de départ pour la recherche des solutions
initial_guess = [1, 1, 1, 1]

# Résolution du système d'équations
solution = fsolve(system, initial_guess)

# Extraire les résultats
x0_solution, x1_solution, x3_solution, x4_solution = solution
x2_solution = V / (x0_solution * x1_solution)

# Affichage des multiplicateurs de Lagrange x3 (p1) et x4 (p2)
multiplicateurs_lagrange = {
    "Multiplicateur de Lagrange p1 (x3)": x3_solution,
    "Multiplicateur de Lagrange p2 (x4)": x4_solution
}

solution_values = {
    "x0": x0_solution,
    "x1": x1_solution,
    "x2": x2_solution,
    "x3 (p1)": x3_solution,
    "x4 (p2)": x4_solution
}

print("Solutions :", solution_values)
print("Multiplicateurs de Lagrange :", multiplicateurs_lagrange)

# %%
