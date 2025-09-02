import numpy as np

"""

"""
def cs_cosamp(y, T_Mat, s, max_iter = 1000, tol = 1e-6):
    """
    CoSaMP algorithm for sparse signal recovery

    Parameters
    ----------
    y : ndarray, shape (n,)
        Measurement vector
    T_Mat : ndarray, shape (n, m)
        Measurement matrix
    s: 	int 
        Number of expected non-zero coefficients

  
    Returns
    -------
    hat_x : ndarray, shape (m,)
        Recovered sparse signal
    """
    m =  T_Mat.shape[1]
    n = len(y)
    r_n = y.copy()                      

    sig_pos_lt = []      
    T_Mat = T_Mat / np.linalg.norm(T_Mat, axis = 0, keep = True)
 
    for _ in range(max_iter):

        #proxy
        product = np.abs(T_Mat.T @ r_n)
        pos = np.argsort(product)[::-1]
        # identify
        sig_pos_cr = pos[:2*s]           

        # step 2: merge supports
        sig_pos = np.union1d(sig_pos_cr, sig_pos_lt)

        # step 3: least squares on merged support
        Aug_t = T_Mat[:, sig_pos]
        aug_x_cr = np.zeros(m)
        # solve (A^T A)^{-1} A^T y
        aug_x_cr[sig_pos] = np.linalg.pinv(Aug_t) @ y

        # step 4: prune to s largest
        pos2 = np.argsort(np.abs(aug_x_cr))[::-1]
        hat_x = np.zeros(m)
        hat_x[pos2[:s]] = aug_x_cr[pos2[:s]]

        # step 5: update support
        sig_pos_lt = pos2[:s]

        # step 6: update residual
        r_n = y - T_Mat @ hat_x

    return hat_x
