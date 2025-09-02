import jax
from jax import random
import jax.numpy as jnp
# Some keys for generating random numbers
key = random.PRNGKey(0)
keys = random.split(key, 4)
from cr.sparse.pursuit import cosamp

# For plotting diagrams
import matplotlib.pyplot as plt
# CR-Sparse modules
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.data as crdata
from cr.nimble.dsp import (
    nonzero_indices,
    nonzero_values,
    largest_indices
)

M = 128
# Ambient dimension
N = 256
# Sparsity level
K = 8

# %%
# The Sparsifying Basis
# ''''''''''''''''''''''''''
Phi = crdict.gaussian_mtx(key, M,N)
print(Phi.shape)
print(crdict.coherence(Phi))
x0, omega = crdata.sparse_normal_representations(key, N, K)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0)
y = Phi @ x0
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(y)


r=y
y_norm_sqr = float(y.T @ y)
r_norm_sqr = y_norm_sqr
print(f"{r_norm_sqr=}")
flags = jnp.zeros(N, dtype=bool)
K2 = 2*K
K3 = K + K2
iterations = 0

solution =  cosamp.matrix_solve_jit(Phi, y, K)
# The support for the sparse solution
I = solution.I
x_I = solution.x_I
print(x_I)
print(I)
print(jnp.setdiff1d(omega, I))
print(solution.r_norm_sqr, solution.iterations)
x = jnp.zeros(N).at[I].set(x_I)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0, label="Original vector")
plt.plot(x, '--', label="Estimated solution")
plt.legend()
plt.show()