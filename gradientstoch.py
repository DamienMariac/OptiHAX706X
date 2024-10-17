#%%
import numpy as np
import matplotlib.pyplot as plt

def cost_function(theta):
    return (theta[0] - 3)**2 + (theta[1] + 2)**2

def gradient(theta):
    grad = np.zeros(2)
    grad[0] = 2 * (theta[0] - 3)  # Dérivée par rapport à theta_0
    grad[1] = 2 * (theta[1] + 2)  # Dérivée par rapport à theta_1
    return grad

# Descente de gradient stochastique
def stochastic_gradient_descent(learning_rate=0.1, iterations=100, random_seed=None):
    np.random.seed(random_seed)
    
    # Initialisation aléatoire des paramètres
    theta = np.random.randn(2)
    theta_history = [theta.copy()]  # Pour stocker les valeurs de theta
    cost_history = [cost_function(theta)]
    
    for i in range(iterations):
        grad = gradient(theta)
        # Mise à jour de theta selon le gradient et le taux d'apprentissage
        theta -= learning_rate * grad
        
        # Stockage des valeurs pour analyse
        theta_history.append(theta.copy())
        cost_history.append(cost_function(theta))
        
        # Affichage des valeurs intermédiaires (facultatif)
        if i % 10 == 0:
            print(f"Iteration {i}: theta = {theta}, cost = {cost_history[-1]}")
    
    return theta, theta_history, cost_history

# Exécution de l'algorithme
theta_final, theta_history, cost_history = stochastic_gradient_descent(learning_rate=0.1, iterations=50)

# Visualisation des résultats
plt.plot(cost_history)
plt.title('Évolution de la fonction de coût au cours des itérations')
plt.xlabel('Itérations')
plt.ylabel('Coût')
plt.show()

# %%
# Importation des bibliothèques nécessaires
from mpl_toolkits.mplot3d import Axes3D

# Descente de gradient stochastique
def stochastic_gradient_descent_3d(learning_rate=0.1, iterations=50, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    
    # Initialisation aléatoire des paramètres
    theta = np.random.randn(2)
    theta_history = [theta.copy()]  # Pour stocker les valeurs de theta
    
    for i in range(iterations):
        grad = gradient(theta)
        # Mise à jour de theta selon le gradient et le taux d'apprentissage
        theta -= learning_rate * grad
        
        # Stockage des valeurs pour analyse
        theta_history.append(theta.copy())
    
    return np.array(theta_history)

# Exécution de l'algorithme
theta_history = stochastic_gradient_descent_3d(learning_rate=0.1, iterations=50)

# Création du meshgrid pour tracer la surface de la fonction de coût
theta0_vals = np.linspace(-1, 5, 100)
theta1_vals = np.linspace(-6, 2, 100)
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
Z = (T0 - 3)**2 + (T1 + 2)**2

# Création du graphique 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface de la fonction de coût
ax.plot_surface(T0, T1, Z, cmap='viridis', alpha=0.7)

# Tracer la trajectoire de la descente de gradient
theta_history = np.array(theta_history)
cost_history = (theta_history[:, 0] - 3)**2 + (theta_history[:, 1] + 2)**2
ax.plot(theta_history[:, 0], theta_history[:, 1], cost_history, 'o-', color='red', label="Descente de gradient")

# Configuration des axes
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel('Coût')
ax.set_title('Descente de gradient stochastique sur la surface de coût')

plt.show()

# %%
