#%%
import numpy as np
import matplotlib.pyplot as plt



def FDC(func, x, h):
    H = np.eye(x.size) * h
    return np.array([(func(x + hi) - func(x - hi)) / 2 for hi in H[:,]])


def FD_GD(func, x, h, rho, max_iter):
    iter_count = 0
    while np.linalg.norm(FDC(func, x, h)) > rho and iter_count < max_iter:
        # Calculate new iterate xn+1 using gradient descent step
        x = x - rho * FDC(func, x, h)
        iter_count += 1
    return x


E = lambda x: [x[0] + x[1] + x[2] - 1]


J = lambda x: x[0] ** 2 - 4 * x[0] - x[0] * x[1] + x[0] * x[2] + x[1] * x[2]

tildJ_alpha = lambda alphas: (lambda x: J(x) + np.sum([a * np.abs(ei) for (a, ei) in zip(alphas, E(x))]))

def Penalisation(func, x, h, rho, max_iter):
    return FD_GD(func, x, h, rho, max_iter)

n = 3
x0 = np.ones(n)
rho = 1e-3
max_iter = 1000
Alphas = np.linspace(0, 30, num=100)
X_alphas = np.array([Penalisation(tildJ_alpha([alpha]), x0, 1e-1, rho, max_iter) for alpha in Alphas])
# print(X_alphas)


print(len(X_alphas))
E_alphas = [E(x)[0] for x in X_alphas]
J_alphas = [J(x) for x in X_alphas]
# plt.plot(E_alphas, J_alphas, 'o')
plt.plot(Alphas, J_alphas, 'o',)
# plt.scatter(E_alphas[0], J_alphas[0], color='red')
plt.scatter(Alphas[0], J_alphas[0], color='red', s=30 )
# plt.xlabel("E alpha")
plt.xlabel("Alpha")
plt.ylabel("J alpha")
plt.tight_layout()
plt.legend()
# plt.figure()
plt.plot(Alphas, [np.linalg.norm(x-np.array([-1,1,-1])) for x in X_alphas], 'o',label="dist to opt")
plt.xlabel("alpha")
plt.ylabel("dist to J theoretical min")
plt.tight_layout()
plt.legend()
# %%
