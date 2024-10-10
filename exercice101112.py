#%%
import numpy as np
from scipy.optimize import fsolve

#j'ai fais par substitution

def equation(x1):
    x2 = x1**3 - 1 
    eq = 3 * x1**2 * x2 + 3 * x1**5 
    return eq


x1_solution = fsolve(equation, 0.8)  # Un point proche de la racine cube de 1/2

x1 = x1_solution[0]
x2 = x1**3 - 1
minimum_J = x1**3 * x2

print("Solution trouvée :")
print("x1 =", x1)
print("x2 =", x2)
print("Minimum de J(x1, x2) =", minimum_J)


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

# Fonction substitué
def equation(x1):
    x2 = x1**3 - 1 
    eq = 3 * x1**2 * x2 + 3 * x1**5 
    return eq


x1_solution = fsolve(equation, 0.8)

x1 = x1_solution[0]
x2 = x1**3 - 1
minimum_J = x1**3 * x2

# Génération des valeurs pour la visualisation
x1_ = np.linspace(-2, 2, 400)
x2_ = x1_**3 - 1
J_ = x1_**3 * x2_

# Création de la figure 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracé de la surface représentant J(x1, x2)
ax.plot(x1_, x2_, J_, label="J(x1, x2)", color='blue')

ax.scatter(x1, x2, minimum_J, color='red', label=f'Minimum: ({x1:.2f}, {x2:.2f}, {minimum_J:.2f})', s=100)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("J")

ax.legend()
plt.show()

# %%
