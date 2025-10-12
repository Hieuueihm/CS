import numpy as np


def cs_iht(y, T_Mat, k=None, itermax=1000, patience=20):
    """
    Iterative Hard Thresholding (IHT) for Compressive Sensing
    y = T_Mat * x,  T_Mat is n-by-m

    Parameters:
        y       : ndarray, measurements (n,)
        T_Mat   : ndarray, sensing matrix (n,m)
        k       : int, sparsity (if None, use len(y)//4)
        itermax : int, maximum iterations
        patience: int, convergence

    Returns:
        hat_x   : ndarray, recovered signal (m,)
    """
    m = T_Mat.shape[1]
    hat_x_tp = np.zeros(m)  # initialization
    T_Mat = T_Mat / np.linalg.norm(T_Mat, axis=0, keepdims=True)

    # L = np.linalg.norm(T_Mat, 2) ** 2
    # u = 1.0 / L
    # print(u)
    u = 0.5
    last_support = set()
    stable_count = 0
    if k is None:
        k = len(y) // 4
    for it in range(itermax):

        x_increase = T_Mat.T @ (y - T_Mat @ hat_x_tp)
        hat_x = hat_x_tp + u * x_increase

        pos = np.argsort(hat_x)[::-1]  # sx giam dan
        hat_x[pos[k:]] = 0
        support = set(np.nonzero(hat_x)[0])
        if support == last_support:
            stable_count += 1
        else:
            stable_count = 0
        last_support = support

        if stable_count >= patience:
            break
        hat_x_tp = hat_x

    return hat_x
