#%%
import numpy as np


def f():
    return 0

def grad_f():
    return 0


def gradient_descent(y, lr=0.1, tol=1e-6, max_iter=10000):
    
    x = np.random.randn(*y.shape)  # Initialisation aléatoire de x (cf Salmon en L3)
    
    for i in range(max_iter):
        grad = grad_f(x, y)
        x_new = x - lr * grad 
        
        # Critère d'arrêt : si la différence est suffisamment petite
        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergence atteinte après {i} itérations.")
            break
        x = x_new
    
    return x
# %%
