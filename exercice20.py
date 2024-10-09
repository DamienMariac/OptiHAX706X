#%%
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def Q0(x):
    return 1

def Q1(x):
    return x

def Q2(x):
    return x**2

def f_x3(x):
    return x**3

def f_sin(x):
    return np.sin(x)

def f_exp(x):
    return np.exp(x)

# f_x3
def J(V, f, Qs):
    a, b, c = V

    P_f = lambda x: a * Qs[0](x) + b * Qs[1](x) + c * Qs[2](x)
    
    integrand = lambda x: (f(x) - P_f(x))**2
    integral, _ = quad(integrand, -1, 1)
    
    return integral

#avec différences finies
def gradient_df(J, V, f, Qs, h=1e-6):
    grad = np.zeros(len(V))
    for i in range(len(V)):
        V_plus = np.array(V, dtype=float)
        V_minus = np.array(V, dtype=float)
        V_plus[i] += h
        V_minus[i] -= h
        grad[i] = (J(V_plus, f, Qs) - J(V_minus, f, Qs)) / (2 * h)
    return grad

# Validation avec Vopt pour f = x^3
V_opt = [0, 3/5, 0]

# Calcul de J pour Vopt
Qs = [Q0, Q1, Q2]
f = f_x3  # Choisir la fonction cible ici

# Vérification du résultat pour Vopt
J_opt = J(V_opt, f, Qs)
print(f"J(Vopt) = {J_opt} (valeur attendue : 4/175 = {4/175})")

# Optimisation de J(V) pour trouver a, b, c optimaux
V_init = [0.0, 0.0, 0.0]  # Initialisation des coefficients
result = minimize(J, V_init, args=(f, Qs), method='BFGS', jac=None)
V_optimized = result.x
print(f"V optimisé : {V_optimized}")

# Calcul du gradient par différences finies à la solution optimisée
grad_fd = gradient_df(J, V_optimized, f, Qs)
print(f"Gradient (différences finies) : {grad_fd}")

# %%
