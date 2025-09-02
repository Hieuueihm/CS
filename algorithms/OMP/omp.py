import numpy as np

def cs_omp(y, T_Mat, s = None, max_iter=1000, tol=1e-6):
    """
    Orthogonal Matching Pursuit (OMP) algorithm
    
    Parameters
    ----------
    y : ndarray
        Measurement vector of shape (n,)
    T_Mat : ndarray
        Measurement matrix of shape (n, m)
        (combination of random matrix and sparse representation basis)
    s: int - optional
        Number of expected non-zero coefficients (sparsity).
        If None, stopping is based on tol instead.
    max_iter: int - optional
        Maximum number of interations
    tol: float - optional
        Stopping criteria
        
    Returns
    -------
    hat_x : ndarray
        Reconstructed signal of shape (m,)
    """
    n = len(y)
    m = T_Mat.shape[1]
    hat_x = np.zeros(m)               
    Aug_t = np.zeros((n, 0))         
    r_n = y.copy()
    pos_array = []
    aug_x = 0
    T_Mat = T_Mat / np.linalg.norm(T_Mat, axis=0, keepdims=True)
    it = 0
    if s is not None:  
        n_iter = s
    else:              
        n_iter = max_iter
    for it in range(n_iter):
        # projection
        it += 1
        product = np.abs(T_Mat.T @ r_n)
        # print(product)
        pos = np.argmax(product)
        pos_array.append(pos)
        
        Aug_t = np.hstack((Aug_t, T_Mat[:, [pos]]))
        T_Mat[:, pos] = 0
        aug_x = np.linalg.pinv(Aug_t) @ y
        r_n = y - Aug_t @ aug_x
        # print(np.linalg.norm(r_n))
        if s is None and np.linalg.norm(r_n) < tol:
            break
    hat_x[pos_array] = aug_x    
    return hat_x