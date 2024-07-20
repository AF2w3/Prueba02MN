import numpy as np
import itertools
from scipy.linalg import lu

# Definición de la matriz A
A = np.array([
    [1, 1, -1, 2],
    [2, 4, 2, 5],
    [1, -1, 1, 7],
    [2, 3, 4, 6]
])

# Función para comprobar si una matriz es no singular (determinante no es cero)
def is_non_singular(matrix):
    return np.linalg.det(matrix) != 0

# Generar todas las posibles matrices de permutación P para una matriz 4x4
n = A.shape[0]
permutations = list(itertools.permutations(range(n)))
all_P_matrices = [np.eye(n)[list(p)] for p in permutations]

# Filtrar matrices de permutación que permiten una descomposición LU válida
valid_P_matrices = []
for P in all_P_matrices:
    PA = P @ A
    try:
        L, U, _ = lu(PA)
        if is_non_singular(L) and is_non_singular(U):
            valid_P_matrices.append(P)
    except np.linalg.LinAlgError:
        pass

# Mostrar todas las matrices de permutación válidas
for idx, P in enumerate(valid_P_matrices):
    print(f"Matriz de permutación válida {idx + 1}:\n{P}\n")