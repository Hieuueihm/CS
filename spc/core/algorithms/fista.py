import numpy as np
import time


def cs_fista(
    y,
    A,
    lambda_=2e-5,
    tol=1e-4,
    max_iter=10000,
    return_info=True,
    ignore_iteration_log=False,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    FISTA (Fast ISTA) for L1-regularized Compressive Sensing

    Solve approximately:
        min_x  0.5 * ||y - A x||_2^2 + lambda_ * ||x||_1

    Parameters
    ----------
    y : ndarray, shape (m,)
        Measurement vector.
    A : ndarray, shape (m, n)
        Measurement matrix.
    lambda_ : float, optional
        L1 regularization parameter.
    tol : float, optional
        Tolerance for stopping (relative change or residual norm).
    max_iter : int, optional
        Maximum number of iterations.
    return_info : bool, optional
        Whether to return AlgorithmInformation.
    ignore_iteration_log : bool, optional
        If True, do not log per-iteration information.

    Returns
    -------
    x_hat : ndarray, shape (n,)
        Recovered signal.
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, n = A.shape

    info = AlgorithmInformation(algorithm_name="fista") if return_info else None
    if info is not None and ignore_iteration_log:
        info.iteration_log = False

    if info is not None:
        info.set_meta("lambda_", lambda_)
        info.set_meta("tol", tol)
        info.set_meta("max_iter", max_iter)

    x_prev = np.zeros(n)
    x_curr = np.zeros(n)
    t_prev = 1.0
    L = np.linalg.norm(A, 2) ** 2
    muy = 1.0 / (L + 1e-12)

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for it in range(1, max_iter + 1):
        t_curr = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        z = x_curr + ((t_prev - 1.0) / t_curr) * (x_curr - x_prev)

        r_z = y - A @ z
        g_z = A.T @ r_z
        u = z + muy * g_z

        x_next = np.sign(u) * np.maximum(np.abs(u) - muy * lambda_, 0.0)

        dx_rel = float(
            np.linalg.norm(x_next - x_curr) / (np.linalg.norm(x_next) + 1e-12)
        )
        res = y - A @ x_next
        resn = float(np.linalg.norm(res))

        if info is not None and info.iteration_log:
            sparsity = int(np.count_nonzero(x_next))
            info.add_iteration(
                residual_norm=resn,
                sparsity=sparsity,
                step_size=muy,
                support=None,
            )

        if dx_rel < tol:
            stop_reason = "dx_rel"
            break
        if resn < tol:
            stop_reason = "residual"
            break

        x_prev = x_curr
        x_curr = x_next
        t_prev = t_curr

    x_hat = x_next
    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.FISTA_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)
        return x_hat, info

    return x_hat
