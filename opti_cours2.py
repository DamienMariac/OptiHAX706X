#%%
import numpy as np

# La fonction
def j(x):
    return 0.5 * (x[0]**2 + 3*x[1]**2) + 3*x[0]*x[1] + 1

# Le gradient
def grad_j(x):
    f1= x[0] + 3*x[1]
    f2 = 6*x[1] + 3*x[0]
    return np.array([f1, f2])

# Méthode de gradient
def gradient_descent(x0, pas=0.01, n=10000):
    x = np.array(x0)
    for i in range(n):
        grad = grad_j(x)
        x = x - pas * grad
    return x


x0 = [1.0, 1.0]  
pas = 0.01  
n = 100  


x_min = gradient_descent(x0, pas, n)
print(f"minimum en x = {x_min} avec j(x) = {j(x_min)}")
# %%

# Autre version avec un critere d'arret

def j(x):
    return 0.5 * (x[0]**2 + 3*x[1]**2) + 3*x[0]*x[1] + 1


def grad_j(x):
    f1= x[0] + 3*x[1]
    f2 = 6*x[1] + 3*x[0]
    return np.array([f1, f2])

def gradient_descent(x0, pas=0.01, n=10000, esp=0.01):
    x = np.array(x0)
    grad = grad_j(x)
    while np.linalg.norm(grad)>esp:
        x = x - pas * grad
    return x


x0 = [1.0, 1.0]  
pas = 0.01  
n = 100 
esp=0.01


x_min = gradient_descent(x0, pas, n, esp)
print(x_min)

# %%
