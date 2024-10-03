#%%
import numpy as np
from scipy.linalg import hilbert


def J(A, B, X):
    return 0.5 * np.dot(X.T, np.dot(A, X)) - np.dot(B.T, X)

def gradient(A, B, X):
    return np.dot(A, X) - B


def gradient_descent(A, B, gam=0.01, eps=1e-8, max_iter=10000):
    X = np.zeros_like(B)
    for i in range(max_iter):
        grad = gradient(A, B, X)
        X_new = X - gam * grad

        if np.linalg.norm(X_new - X) < eps:
            break
        X = X_new
    return X



n = 10 
A = hilbert(n)
x_target = np.ones(n)  
B = np.dot(A, x_target)  
X_init = np.zeros(n)
X_solution = gradient_descent(A, B, X_init)
erreur = np.linalg.norm(X_solution - x_target)


print("Solution X trouvée :")
print(X_solution)
print("Solution cible x_target :")
print(x_target)
print(f"Erreur par rapport à la solution cible : {erreur}")


# %%
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

# %%
