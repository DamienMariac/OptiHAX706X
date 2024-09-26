#%%
import numpy as np

def methode_grad_fini(f, x, h=1e-5):

    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.copy(x)
        x_h[i] += h  
        grad[i] = (f(x_h) - f(x)) / h 
    return grad

def gradient_descent(f, x_init, learning_rate=0.01, tolerance=1e-6, max_iters=1000, h=1e-5):
   
    x = np.copy(x_init)
    history = [x]

    for i in range(max_iters):
        grad = methode_grad_fini(f, x, h)
        x_new = x - learning_rate * grad

        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
        history.append(x)

    return x, history


def example_function(x):
    return x[0]**2 + x[1]**2


x_init = np.array([2.0, 3.0])  

minimum, history = gradient_descent(example_function, x_init, learning_rate=0.1)

print(minimum)
print(history)

# %%
