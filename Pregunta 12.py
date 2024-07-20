import numpy as np
import matplotlib.pyplot as plt

def gauss_jacobi(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    trajectory = [x.copy()]
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, trajectory

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    trajectory = [x.copy()]
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, trajectory

# Definición del sistema de ecuaciones
A = np.array([
    [1, 1],
    [-2, 5]
])

b = np.array([7, 0])

# Inicialización
x0 = np.array([0, 0])
tol = 1e-10
max_iter = 100

# Solución con el método de Gauss-Jacobi
x_jacobi, trajectory_jacobi = gauss_jacobi(A, b, x0, tol, max_iter)

# Solución con el método de Gauss-Seidel
x_seidel, trajectory_seidel = gauss_seidel(A, b, x0, tol, max_iter)

# Dibujo de las trayectorias
plt.figure(figsize=(10, 6))
plt.plot(*zip(*trajectory_jacobi), 'b-o', label='Trayectoria Gauss-Jacobi')
plt.plot(*zip(*trajectory_seidel), 'r-x', label='Trayectoria Gauss-Seidel')
plt.scatter([5], [2], c='orange', label='Solución')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True)
plt.title('Trayectorias de los métodos iterativos')
plt.show()