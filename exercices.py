"""
LES CODES NE SONT PAS TOUS BON, IL FAUT FAIRE ATTENTION

"""

#################################EXO 3
# %%
import numpy as np
import matplotlib.pyplot as plt

# f(x) = <x, y> exp(-||x||^2)
def f(x, y):
    scalar_product = np.dot(x, y)
    norm_squared = np.linalg.norm(x)**2
    return scalar_product * np.exp(-norm_squared)


y = np.array([1, 0])
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)


Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = f(x, y)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(X1, X2, Z, cmap='hot', edgecolor='none')


ax.set_title("Graphique 3D de f(x)", fontsize=12)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
plt.show()

#%%
############################################## EXO 3 VARIANTE

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    scalar_product = np.dot(x, y)
    norm_squared = np.linalg.norm(x)**2
    return scalar_product * np.exp(-norm_squared)

def grad_f(x, y):
    exp_term = np.exp(-np.linalg.norm(x)**2)
    gradient = exp_term * (y - 2 * np.dot(x, y) * x)
    return gradient


x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xi = np.array([X[i, j], Y[i, j]])
        yi = np.array([1.0, 1.0]) 
        Z[i, j] = f(xi, yi)


fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Fonction $f(x, y)$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')


x0 = np.array([1.0, 1.0])
y = np.array([1.0, 1.0])
learning_rate = 0.1
iterations = 50
values = []

x_iter = x0.copy()
for i in range(iterations):
    values.append(f(x_iter, y))
    grad = grad_f(x_iter, y)
    x_iter = x_iter - learning_rate * grad


ax2 = fig.add_subplot(122)
ax2.plot(range(iterations), values, marker='o')
ax2.set_title('Valeur de $f(x, y)$ vs Itérations')
ax2.set_xlabel('Itérations')
ax2.set_ylabel('Valeur de $f(x, y)$')

plt.tight_layout()
plt.show()


# %%
##############################################exercice 4

import numpy as np

def methode_grad_fini(f, x, h=1e-5):

    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.copy(x)
        x_h[i] += h  
        grad[i] = (f(x_h) - f(x)) / h 
    return grad

def gradient_descent(f, x_init, learning_rate=0.01, tolerance=1e-6, max_iters=1000, h=1e-5):
   
    x = np.copy(x_init)
    history = [x]

    for i in range(max_iters):
        grad = methode_grad_fini(f, x, h)
        x_new = x - learning_rate * grad

        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
        history.append(x)

    return x, history


def example_function(x):
    return x[0]**2 + x[1]**2


x_init = np.array([2.0, 3.0])  

minimum, history = gradient_descent(example_function, x_init, learning_rate=0.1)

print(minimum)
print(history)
# %%
########################################### EXO5

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

def gradient(A, B, X):
    return np.dot(A, X) - B

def gradient_descent(A, B, a=0.01, eps=1e-10, iter=10000):
    X = np.zeros_like(B)  
    for i in range(iter):
        grad = gradient(A, B, X)
        X_new = X - a * grad
        
        
        if np.linalg.norm(X_new - X) < eps:
            break
       
        X = X_new
    
    return X


n = 10
A = hilbert(n) 
B = np.dot(A, np.ones(n))


X_approx = gradient_descent(A, B, a=0.01)


X_theoretical = np.ones(n)


plt.figure(figsize=(10, 6))
plt.plot(range(n), X_approx, label="descente grad")
plt.plot(range(n), X_theoretical, label="soluce")

plt.xlabel("Indice du vecteur")
plt.ylabel("Valeur du coefficient")
plt.legend()
plt.show()

#%%
###########################################EXO 6
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

def J(A, B, X):
    return 1/2 * np.dot(np.dot(A, X), X) - np.dot(B, X)

def gradient(A, B, X):
    return np.dot(A, X) - B

n = 20

def gradient_descent_mieux(A, B, eps=1e-3, iter=1000):


    X = np.ones(n) * 3
    grad1 = gradient(A, B, X)
    J0 = J(A, B, X)
    i = 1
    listero = [] 

    while (i < iter and np.linalg.norm(gradient(A, B, X)) / np.linalg.norm(grad1) > eps):

        ro = np.dot(gradient(A, B, X), gradient(A, B, X)) / np.dot(np.dot(A, gradient(A, B, X)), gradient(A, B, X))

        X_new = X - ro * gradient(A, B, X)
        
        if np.linalg.norm(X_new - X) < eps:
            break
        
        X = X_new
        i += 1
        listero.append(ro) 

    print(i)
    
    return X, listero


A = hilbert(n) 
B = np.dot(A, np.ones(n))
X_approx, listero = gradient_descent_mieux(A, B)
X_theo = np.ones(n)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))


axs[0].plot(range(n), X_approx, label="descente grad")
axs[0].plot(range(n), X_theo, label="soluce")
axs[0].set_xlabel("Indice")
axs[0].set_ylabel("coefficient")
axs[0].legend()
axs[0].set_title("Approximation vs solution")


axs[1].plot(range(1, len(listero) + 1), listero, label="ro")
axs[1].set_xlabel("Itération")
axs[1].set_ylabel("Pas (ro)")
axs[1].legend()
axs[1].set_title("pas successif")

plt.show()

#%%

################# EXO 20

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

def J(V, f, Qs):
    a, b, c = V

    P_f = lambda x: a * Qs[0](x) + b * Qs[1](x) + c * Qs[2](x)
    
    integrand = lambda x: (f(x) - P_f(x))**2
    integral, _ = quad(integrand, -1, 1)
    
    return integral

def gradient_df(J, V, f, Qs, h=1e-6):
    grad = np.zeros(len(V))
    for i in range(len(V)):
        V_plus = np.array(V, dtype=float)
        V_minus = np.array(V, dtype=float)
        V_plus[i] += h
        V_minus[i] -= h
        grad[i] = (J(V_plus, f, Qs) - J(V_minus, f, Qs)) / (2 * h)
    return grad


V_opt = [0, 3/5, 0]


Qs = [Q0, Q1, Q2]
f = f_x3  


J_opt = J(V_opt, f, Qs)
print(f"J(Vopt) = {J_opt} (valeur attendue : 4/175 = {4/175})")

V_init = [0.0, 0.0, 0.0]  
result = minimize(J, V_init, args=(f, Qs), method='BFGS', jac=None)
V_optimized = result.x
print(f"V optimisé : {V_optimized}")

grad_fd = gradient_df(J, V_optimized, f, Qs)
print(f"Gradient (différences finies) : {grad_fd}")
