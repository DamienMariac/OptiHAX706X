#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

def J(A,B,X):
    return 1/2*np.dot(np.dot(A,X),X)-np.dot(B,X)

def gradient(A, B, X):
    return np.dot(A, X) - B

n = 20

def gradient_descent_mieux(A, B, eps=1e-3, iter=1000):
    

    X = np.ones(n) * 3
    grad1=gradient(A,B,X)
    J0=J(A,B,X)
    i=1

    while (i<iter and np.linalg.norm(gradient(A,B,X))/np.linalg.norm(grad1) >eps):

        ro = np.dot(gradient(A, B, X), gradient(A, B, X)) / np.dot(np.dot(A, gradient(A, B, X)), gradient(A, B, X))

        X_new = X - ro * gradient(A,B,X)
        
        if np.linalg.norm(X_new - X) < eps:
            break
       
        X = X_new
        i+=1
    print(i)
    
    return X


A = hilbert(n) 
B = np.dot(A, np.ones(n))
X_approx = gradient_descent_mieux(A, B)
X_theo = np.ones(n)


plt.figure(figsize=(10, 6))
plt.plot(range(n), X_approx, label="descente grad")
plt.plot(range(n), X_theo, label="soluce")

plt.xlabel("Indice du vecteur")
plt.ylabel("Valeur du coefficient")
plt.legend()
plt.show()


#Il faut tracer le produit scalaire des ro successif en fonction du nombre d'itération
# %%
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


# %%
