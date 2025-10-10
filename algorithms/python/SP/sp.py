import numpy as np

def cs_sp(y, T_Mat, k=None, itermax=50, tol=1e-4):
    """
    Subspace Pursuit (SP) for Compressive Sensing
    y = T_Mat * x,  T_Mat is n-by-m

    Parameters:
        y       : ndarray, measurements (n,)
        T_Mat   : ndarray, sensing matrix (n,m)
        k       : int, sparsity (if None, default = len(y)//4)
        itermax : int, maximum number of iterations
        tol     : float, residual tolerance for stopping

    Returns:
        hat_x   : ndarray, recovered sparse signal (m,)
    """
    n, m = T_Mat.shape
    if k is None:
        k = n // 4                  

    r_n = y.copy()                  # initial residual
    sig_pos_lt = []                 # previous support
    hat_x = np.zeros(m)
    T_Mat = T_Mat / np.linalg.norm(T_Mat, axis = 0, keepdims = True)
    # print("a")
    for _ in range(itermax):
        # correlation step
        product = np.abs(T_Mat.T @ r_n)
        pos = np.argsort(product)[::-1]
        sig_pos_cr = pos[:k]

        # merge supports
        sig_pos = np.union1d(sig_pos_cr, sig_pos_lt).astype(int)

        # least squares on selected columns
        Aug_t = T_Mat[:, sig_pos]
        aug_x_cr = np.zeros(m)
        aug_x_cr[sig_pos] = np.linalg.pinv(Aug_t) @ y

        # keep best k entries
        pos2 = np.argsort(np.abs(aug_x_cr))[::-1]
        hat_x = np.zeros(m)
        hat_x[pos2[:k]] = aug_x_cr[pos2[:k]]

        # update support and residual
        sig_pos_lt = pos2[:k]
        r_n = y - T_Mat @ hat_x

        # stopping criterion
        if np.linalg.norm(r_n) < tol:
            break


    return hat_x

