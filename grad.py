# %%
import numpy as np

def fonction(x):
    return x**2

def gradient(x):
    return 2*x


def descente_de_gradient(x0, a, n):
    x = x0
    historique = [x] 

    for i in range(n):
        grad = gradient(x)
        x = x - a * grad  
        historique.append(x)  

        print(f"Iteration {i+1}: x = {x}, f(x) = {fonction(x)}")

    return x, historique


x0 = 10  
a = 0.1  
n = 50  


x_final, historique = descente_de_gradient(x0, a, n)

print(x_final)

# %%
