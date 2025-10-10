import numpy as np

def cs_cosamp(y, T_Mat, k, max_iter = 50, tol = 1e-9):
    """
    CoSaMP algorithm for sparse signal recovery

    Parameters
    ----------
    y : ndarray, shape (n,)
        Measurement vector
    T_Mat : ndarray, shape (n, m)
        Measurement matrix
    k: 	int 
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
    T_Mat = T_Mat / np.linalg.norm(T_Mat, axis = 0, keepdims = True)
    it = 0
    for _ in range(max_iter):
        it += 1
        #proxy
        product = np.abs(T_Mat.T @ r_n)
        pos = np.argsort(product)[::-1]
        # identify 2k large component
        sig_pos_cr = pos[:2*k]           

        sig_pos = np.union1d(sig_pos_cr, sig_pos_lt).astype(int)

        Aug_t = T_Mat[:, sig_pos]
        aug_x_cr = np.zeros(m)
        x_temp = np.linalg.pinv(Aug_t) @ y
        aug_x_cr[sig_pos] = x_temp

        pos2 = np.argsort(np.abs(aug_x_cr))[::-1]
        hat_x = np.zeros(m)
        hat_x[pos2[:k]] = aug_x_cr[pos2[:k]]

        sig_pos_lt = pos2[:k]

        r_n = y - T_Mat @ hat_x
        # print(np.linalg.norm(r_n))
        if np.linalg.norm(r_n) < tol:
        	break

    return hat_x, it
