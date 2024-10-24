# %%
import numpy as np
from scipy.linalg import hilbert

n = 10
A = hilbert(n)
b = np.ones(n)
x = np.zeros(n)  
lambda_ = 0     

alpha = 0.01  
beta = 0.01   
max_iter = 1000  


def projection_l1(x):
    abs_x = np.abs(x)
    
    if np.sum(abs_x) <= 1:
        return x
    
    u = -np.sort(-abs_x)  
    sv = np.cumsum(u) - 1
    rho = np.where(u > sv / (np.arange(1, len(x) + 1)))[0][-1]
    theta = sv[rho] / (rho + 1)
    
    return np.sign(x) * np.maximum(abs_x - theta, 0)


##########################################################

for i in range(max_iter):
    grad_x = A.T @ (A @ x - b) + lambda_ * np.sign(x)  
    x = projection_l1(x - alpha * grad_x)  

 
    lambda_ = lambda_ + beta * (np.sum(np.abs(x)) - 1)

    if np.linalg.norm(grad_x) < 1e-6:
        print(f"Convergence atteinte à l'itération {i}")
        break

print("Solution primal x:", x)
print("Multiplicateur de Lagrange lambda:", lambda_)



# %%