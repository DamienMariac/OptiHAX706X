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
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
# from mpl_toolkits.mplot3d import Axes3D

# # Fonction substitué
# def equation(x1):
#     x2 = x1**3 - 1 
#     eq = 3 * x1**2 * x2 + 3 * x1**5 
#     return eq


# x1_solution = fsolve(equation, 0.8)

# x1 = x1_solution[0]
# x2 = x1**3 - 1
# minimum_J = x1**3 * x2

# # Génération des valeurs pour la visualisation
# x1_ = np.linspace(-2, 2, 400)
# x2_ = x1_**3 - 1
# J_ = x1_**3 * x2_

# # Création de la figure 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Tracé de la surface représentant J(x1, x2)
# ax.plot(x1_, x2_, J_, label="J(x1, x2)", color='blue')

# ax.scatter(x1, x2, minimum_J, color='red', label=f'Minimum: ({x1:.2f}, {x2:.2f}, {minimum_J:.2f})', s=100)

# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("J")

# ax.legend()
# plt.show()

#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def J(x):
    return x[0]**2 - 4*x[0] - x[0]*x[1] + x[0]*x[2] + x[1]*x[2]

def h(x):
    return x[0] + x[1] + x[2] - 1

def penalized_function(x, mu):
    return J(x) + mu * h(x)**2

def solve_with_penalty(x0, mu_values):
    solutions = []
    for mu in mu_values:
        result = minimize(penalized_function, x0, args=(mu,), method='trust-constr')
        solutions.append((result.x, J(result.x), h(result.x)))
    return solutions

x0 = np.array([1.0, -1.0, 1.0])

mu_values = np.logspace(-2, 3, 100) 

solutions = solve_with_penalty(x0, mu_values)

J_values = [sol[1] for sol in solutions]  
h_values = [sol[2] for sol in solutions]  
mu_log_values = np.log10(mu_values)  

x_opt = solutions[-1][0]
J_opt = solutions[-1][1]
h_opt = solutions[-1][2]

print("Solution optimale : x =", x_opt)
print("Valeur optimale de J(x) =", J_opt)
print("Vérification de la contrainte h(x) =", h_opt)



# %%


################################### Resultat à partir de Lagrange ###############################

import numpy as np
from scipy.optimize import fsolve

# Fonction objectif J(x)
def J(x):
    return x[0]**2 - 4*x[0] - x[0]*x[1] + x[0]*x[2] + x[1]*x[2]

# Contrainte h(x)
def h(x):
    return x[0] + x[1] + x[2] - 1

# Dérivées partielles de Lagrange (fonction de Lagrange)
def lagrange_system(vars):
    x1, x2, x3, lambda_ = vars
    # Gradient de Lagrange
    dL_dx1 = 2*x1 - 4 - x2 + x3 + lambda_
    dL_dx2 = -x1 + x3 + lambda_
    dL_dx3 = x1 + x2 + lambda_
    dL_dlambda = x1 + x2 + x3 - 1  # Contrainte h(x)
    return [dL_dx1, dL_dx2, dL_dx3, dL_dlambda]

# Point initial de départ
x0 = [0.0, 0.0, 0.0, 0.0]

# Résolution du système non linéaire avec fsolve
solution = fsolve(lagrange_system, x0)

# Résultats
x1_opt, x2_opt, x3_opt, lambda_opt = solution

print("Solution optimale : x1 =", x1_opt, ", x2 =", x2_opt, ", x3 =", x3_opt)
print("Valeur optimale de J(x) =", J([x1_opt, x2_opt, x3_opt]))
print("Vérification de la contrainte h(x) =", h([x1_opt, x2_opt, x3_opt]))

# %%
