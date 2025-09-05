from ista import cs_ista
from fista import cs_fista
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# giả sử bạn đã có cs_fista và cs_ista từ trước

# Tham số
N = 1024
M = 512
K = 10

# Tạo tín hiệu sparse
x = np.zeros((N, 1))
T = 5 * np.random.randn(K, 1)
index_k = np.random.permutation(N)
x[index_k[:K]] = T

A = np.random.randn(M, N)
A = np.sqrt(1/M) * A

Q, _ = np.linalg.qr(A.T)
A = Q.T

# Đo y
y = A @ x

# Khôi phục bằng FISTA và ISTA
x_rec1, error1 = cs_fista(y, A, lambda_=5e-3, epsilon=1e-4, itermax=5000)
x_rec2, error2 = cs_ista(y, A, lambda_=5e-3, epsilon=1e-4, itermax=5000)

# Plot error norm
plt.figure()
plt.plot(error1[:,1], 'r-', label='fista')
plt.plot(error2[:,1], 'b-', label='ista')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Convergence of ISTA vs FISTA")
plt.grid(True)

# Plot signal recovery
plt.figure()
plt.plot(range(N), x, 'r', label='original')
plt.plot(range(N), x_rec1, 'g*', label='fista')
plt.plot(range(N), x_rec2, 'b^', label='ista')
plt.legend()
plt.title("Signal reconstruction")
plt.grid(True)

plt.show()