import numpy as np
import time


def cs_omp(
    y, A, k=None, max_iter=1000, tol=1e-6, return_info=True, ignore_iteration_log=False
):
    from utils.algorithm_information import AlgorithmInformation

    """
    Orthogonal Matching Pursuit (OMP) algorithm

    Parameters
    ----------
    y : ndarray, shape (m,)
        Measurement vector.
    A : ndarray, shape (m, n)
        Measurement matrix.
    k : int, optional
        Expected sparsity level (số phần tử khác 0).
        Nếu None thì dừng theo tol hoặc max_iter.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Stopping criterion on ||residual||_2.
    return_info: bool, optional
        Whether to return additional information about the algorithm's execution.

    Returns
    -------
    hat_x : ndarray, shape (n,)
        Reconstructed sparse signal.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    info = AlgorithmInformation(algorithm_name="omp") if return_info else None
    if info is not None and ignore_iteration_log:
        info.iteration_log = False
    m, n = A.shape

    hat_x = np.zeros(n)
    residual = y.copy()  # r_0 = y
    support = []
    if k is not None:
        n_iter = min(k, max_iter)
    else:
        n_iter = max_iter
    if info is not None:
        info.set_meta("max_iter", n_iter)
        info.set_meta("tol", tol)
    x_S = None
    t0 = time.time()
    stop_reason = "max_iter_reached"
    for _ in range(n_iter):
        #  c = A^T r
        correlations = A.T @ residual  # shape (n,)

        pos = int(np.argmax(np.abs(correlations)))

        if pos not in support:
            support.append(pos)

        A_S = A[:, support]  # shape (m, |S|)

        x_S, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)

        residual = y - A_S @ x_S
        if info is not None and info.iteration_log:
            info.add_iteration(
                residual_norm=np.linalg.norm(residual),
                sparsity=len(support),
                support=support.copy(),
            )

        if np.linalg.norm(residual) < tol:

            stop_reason = "tol_reached"
            break
        if k is not None and len(support) >= k:
            stop_reason = "sparsity_level_reached"
            break
    t1 = time.time()
    if x_S is not None:
        hat_x[support] = x_S.ravel()

    if return_info:
        if info is not None:
            info.set_stop_reason(
                AlgorithmInformation.OMP_STOP_INFORMATION + ":" + stop_reason
            )
            info.set_time(t1 - t0)
        print("OMP finished:", stop_reason)
        return hat_x, info
    return hat_x, _
