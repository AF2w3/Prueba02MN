import logging
from sys import stdout
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


# Función de descomposición LU con pivoteo parcial
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A con pivoteo parcial.

    ## Parameters
    `A`: matriz cuadrada de tamaño n-by-n.

    ## Return
    `L`: matriz triangular inferior.
    `U: matriz triangular superior. Se obtiene de la matriz ``A` después de aplicar la eliminación gaussiana.
    """
    A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    P = np.eye(n)

    for i in range(n):
        max_index = np.argmax(np.abs(A[i:n, i])) + i
        if i != max_index:
            A[[i, max_index]] = A[[max_index, i]]
            P[[i, max_index]] = P[[max_index, i]]
            if i > 0:
                L[[i, max_index], :i] = L[[max_index, i], :i]
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]
            L[j, i] = m
        logging.info(f"\n{A}")
    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")
    return L, A


# Función para calcular el determinante
def calc_determinante(A: list[list[float]]) -> float:
    """Función que calcula el determinante usando la descomposición LU

    ## Parameters
    `A`: Matriz cuadrada de tamaño n x n

    ## Return
    `detA`: Determinante de la matriz A
    """
    # Convertir la lista a un array de numpy
    A = np.array(A, dtype=float)

    # Obtener las matrices L y U de la descomposición LU
    L, U = descomposicion_LU(A)

    # El determinante es el producto de los elementos diagonales de U
    detA = np.prod(np.diag(U))

    return detA


# Matriz A dada en el ejercicio
A = [
    [-4, -2, -4, -4, -2, 5, 5, 1],
    [2, 1, 0, 0, -3, 0, -1, -5],
    [0, 1, 0, 2, 2, 4, 2, 4],
    [-1, -4, -1, 0, 5, 0, 1, 0],
    [0, 4, -4, 0, 3, 0, 4, 5],
    [-1, 2, -1, -1, 1, 3, -1, 1],
    [-4, -2, -3, -1, 3, 1, -4, 5],
    [-4, -2, -3, -1, -1, 1, 1, -3]
]

# Calcular el determinante
detA = calc_determinante(A)
print(f"El determinante de A es: {detA:.9g}")