import numpy as np
import time


def cs_ista(
    y,
    A,
    lambda_=2e-5,
    tol=1e-4,
    max_iter=1000,
    return_info=True,
    ignore_iteration_log=False,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    ISTA (Iterative Soft Thresholding Algorithm) for L1-regularized CS

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
        Tolerance for stopping (applied to both relative change and residual norm).
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

    info = AlgorithmInformation(algorithm_name="ista") if return_info else None
    if info is not None:
        if ignore_iteration_log:
            info.iteration_log = False
        info.set_meta("lambda_", lambda_)
        info.set_meta("tol", tol)
        info.set_meta("max_iter", max_iter)

    x_prev = np.zeros(n)

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for _ in range(1, max_iter + 1):
        r = y - A @ x_prev
        g = A.T @ r

        Ag = A @ g
        num = g @ g
        den = Ag @ Ag + 1e-12
        alpha = float(num / den)

        x_tmp = x_prev + alpha * g

        x_hat = np.sign(x_tmp) * np.maximum(np.abs(x_tmp) - alpha * lambda_, 0.0)

        dx_rel = float(np.linalg.norm(x_hat - x_prev) / (np.linalg.norm(x_hat) + 1e-12))
        res = y - A @ x_hat
        resn = float(np.linalg.norm(res))

        if info is not None and info.iteration_log:
                    sparsity = int(np.count_nonzero(x_hat))

            info.add_iteration(
                residual_norm=resn,
                sparsity=sparsity,
                step_size=alpha,
                support=None,
            )

        if dx_rel < tol:
            stop_reason = "dx_rel"
            break
        if resn < tol:
            stop_reason = "residual"
            break

        x_prev = x_hat

    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.ISTA_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)
        return x_hat, info

    return x_hat
