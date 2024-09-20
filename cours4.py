import numpy as np

def methode_grad_fini(f, x, h=1e-5):
    """
    
    :param f: Fonction à minimiser.
    :param x: Point où calculer le gradient (vecteur).
    :param h: Petite variation pour l'approximation.
    :return: Gradient de f au point x.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.copy(x)
        x_h[i] += h  # Incrémente la i-ème composante de h
        grad[i] = (f(x_h) - f(x)) / h  # Différence finie
    return grad

def gradient_descent(f, x_init, learning_rate=0.01, tolerance=1e-6, max_iters=1000, h=1e-5):
    """
    Minimisation de f en utilisant la descente de gradient avec approximation du gradient par différence finie.
    
    :param f: Fonction à minimiser.
    :param x_init: Point initial pour la descente de gradient (vecteur).
    :param learning_rate: Taux d'apprentissage (eta).
    :param tolerance: Critère de convergence.
    :param max_iters: Nombre maximal d'itérations.
    :param h: Petite variation pour l'approximation du gradient.
    :return: Le minimum trouvé et l'historique des points visités.
    """
    x = np.copy(x_init)
    history = [x]  # Pour stocker l'évolution des points

    for i in range(max_iters):
        grad = methode_grad_fini(f, x, h)
        x_new = x - learning_rate * grad

        # Critère d'arrêt : si le changement est inférieur à la tolérance
        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
        history.append(x)

    return x, history

# Exemple d'utilisation
def example_function(x):
    """
    Exemple de fonction à minimiser : f(x) = x_0^2 + x_1^2
    C'est une parabole dans 2 dimensions avec un minimum global en (0, 0).
    """
    return x[0]**2 + x[1]**2

# Point initial
x_init = np.array([2.0, 3.0])  # Départ de (2, 3)

# Appel de la descente de gradient
minimum, history = gradient_descent(example_function, x_init, learning_rate=0.1)

# Résultats
print("Le minimum trouvé est :", minimum)
print("Historique des points visités :", history)
