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


# %%
import numpy as np

def f(x, y):
    scalar_product = np.dot(x, y)
    norm_squared = np.linalg.norm(x)**2
    return scalar_product * np.exp(-norm_squared)


def grad_f(x, y):
    exp_term = np.exp(-np.linalg.norm(x)**2)
    gradient = exp_term * (y - 2 * np.dot(x, y) * x)
    return gradient


def gradient_descent(y, lr=0.1, tol=1e-6, max_iter=1000):
    x = np.random.randn(*y.shape)
    for i in range(max_iter):
        grad = grad_f(x, y)
        x_new = x - lr * grad t

        if np.linalg.norm(x_new - x) < tol:
            print(i)
            break
        x = x_new
    return x


y = np.array([1, 0]) 


min_x = gradient_descent(y, lr=0.1)

print("Le minimum local de la fonction est atteint en x =", min_x)
print("La valeur de la fonction à ce minimum est f(x) =", f(min_x, y))

#%%
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x**2 + y**2


def gradient(x, y):
    return np.array([2*x, 2*y])


def gradient_descent(starting_point, learning_rate, iterations):
    x, y = starting_point
    points = [(x, y)] 
    values = [f(x, y)] 
    for _ in range(iterations):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        points.append((x, y))  
        values.append(f(x, y))  
    return np.array(points), values


starting_point = np.array([4.0, 3.0])  #
learning_rate = 0.1 
iterations = 50  


points, values = gradient_descent(starting_point, learning_rate, iterations)


x_vals = points[:, 0]
y_vals = points[:, 1]

t
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
x, y = np.meshgrid(x, y)
z = f(x, y)

fig = plt.figure(figsize=(12, 6))


ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis', alpha=0.6)
ax1.plot(x_vals, y_vals, f(x_vals, y_vals), color='r', marker='o', markersize=5, label="Descente de Gradient")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x, y) = x^2 + y^2')
ax1.set_title('Descente de Gradient sur f(x, y)')
ax1.legend()


ax2 = fig.add_subplot(122)
ax2.plot(range(len(values)), values, marker='o', color='b')
ax2.set_xlabel('Nombre d\'itérations')
ax2.set_ylabel('Valeur de la fonction f(x, y)')
ax2.set_title('Valeur de f(x, y) = x^2 + y^2 en fonction des itérations')
ax2.grid(True)

plt.tight_layout()
plt.show()


# %%
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
