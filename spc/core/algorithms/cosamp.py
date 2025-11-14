import numpy as np
import time


def cs_cosamp(
    y,
    A,
    k,
    max_iter=50,
    tol=1e-9,
    return_info=True,
    ignore_iteration_log=False,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    CoSaMP algorithm for sparse signal recovery

    Parameters
    ----------
    y : ndarray, shape (m,)
        Measurement vector.
    A : ndarray, shape (m, n)
        Measurement matrix.
    k : int
        Expected sparsity level (number of non-zero coefficients).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Stopping tolerance on ||residual||_2.
    return_info: bool, optional
        Whether to return additional information about the algorithm's execution.
    ignore_iteration_log : bool, optional
        If True, do not log per-iteration information.

    Returns
    -------
    hat_x : ndarray, shape (n,)
        Recovered sparse signal.
    """

    info = AlgorithmInformation(algorithm_name="cosamp") if return_info else None
    if info is not None and ignore_iteration_log:
        info.iteration_log = False

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    m, n = A.shape

    if k is None:
        k = m // 4

    if info is not None:
        info.set_meta("max_iter", max_iter)
        info.set_meta("tol", tol)
        info.set_meta("k", k)

    hat_x = np.zeros(n)
    residual = y.copy()
    support_prev = np.array([], dtype=int)

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for _ in range(1, max_iter + 1):
        #  c = A^T r
        proxy = A.T @ residual  # shape (n,)

        idx_sorted = np.argsort(np.abs(proxy))
        omega = idx_sorted[-2 * k :]

        candidate_support = np.union1d(omega, support_prev).astype(int)

        A_S = A[:, candidate_support]
        b_S, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)

        idx_sorted_b = np.argsort(np.abs(b_S))
        idx_k = idx_sorted_b[-k:]
        new_support = candidate_support[idx_k]

        x_temp = np.zeros(n)
        x_temp[new_support] = b_S[idx_k]

        residual = y - A @ x_temp
        r_norm = np.linalg.norm(residual)

        if info is not None and info.iteration_log:
            info.add_iteration(
                residual_norm=r_norm,
                sparsity=int(np.count_nonzero(x_temp)),
                support=new_support.copy(),
            )

        # 9. Update support
        support_prev = new_support

        # 10. Điều kiện dừng
        if r_norm < tol:
            stop_reason = "tol_reached"
            break

    t1 = time.time()
    hat_x = x_temp

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.COSAMP_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)
        return hat_x, info

    return hat_x
