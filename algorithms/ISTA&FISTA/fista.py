import numpy as np

import numpy as np

def cs_fista(y, A, lambda_=2e-5, epsilon=1e-4, itermax=10000):

    N = A.shape[1]
    errors = []

    x_0 = np.zeros((N, 1))
    x_1 = np.zeros((N, 1))
    t_0 = 1.0


    for i in range(itermax):
        t_1 = (1 + np.sqrt(1 + 4 * t_0**2)) / 2

        g_1 = A.T @ (y - A @ x_1)
        Ag = A @ g_1
        alpha = (g_1.T @ g_1) / (Ag.T @ Ag + 1e-12) 
        alpha = float(alpha) 

        # alpha = 1.0
        # extrapolation
        z_2 = x_1 + ((t_0 - 1) / t_1) * (x_1 - x_0)

        # gradient step
        z_2 = z_2 + alpha * (A.T @ (y - A @ z_2))
        # soft-thresholding
        x_2 = np.sign(z_2) * np.maximum(np.abs(z_2) - alpha * lambda_, 0)
        err1 = np.linalg.norm(x_2 - x_1) / (np.linalg.norm(x_2) + 1e-12)
        err2 = np.linalg.norm(y - A @ x_2)
        errors.append([err1, err2])

        if err1 < epsilon or err2 < epsilon:
            break
        else:
            x_0 = x_1
            x_1 = x_2
            t_0 = t_1

    return x_2, np.array(errors)
