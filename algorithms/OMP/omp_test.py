import numpy as np
from omp import cs_omp
# --- Ma trận đo A (3x4) ---
A = np.array([
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1]
], dtype=float)

# --- Tín hiệu sparse gốc ---
x_true = np.array([0, 3, 0, -2], dtype=float)

# --- Vector đo ---
y = A @ x_true
print("y: ")
print(y)
x_rec = cs_omp(y, A.copy())

print("Tín hiệu gốc:", x_true)
print("Vector đo y:", y)
print("Tín hiệu khôi phục:", x_rec)
