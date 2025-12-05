import numpy as np
import time
from core.metrics import psnr


def cs_ista(
    y,
    A,
    lambda_=2e-5,
    tol=1e-4,
    max_iter=1000,
    return_info=True,
    ignore_iteration_log=False,
    measure_memory=False,
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
    measure_memory : bool, optional
        If True, estimate static and peak memory usage based on array sizes.

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

    # --------- MEMORY: static + peak (ước lượng) ----------
    if measure_memory:
        # static: A, y, x_prev
        static_bytes = A.nbytes + y.nbytes + x_prev.nbytes
        peak_bytes = static_bytes
    else:
        static_bytes = None
        peak_bytes = None
    # ------------------------------------------------------

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for _ in range(1, max_iter + 1):
        # Gradient-like step
        r = y - A @ x_prev  # residual
        g = A.T @ r  # gradient

        Ag = A @ g
        num = g @ g
        den = Ag @ Ag + 1e-12
        muy = float(num / den)

        x_tmp = x_prev + muy * g

        x_hat = np.sign(x_tmp) * np.maximum(np.abs(x_tmp) - muy * lambda_, 0.0)

        dx_rel = float(np.linalg.norm(x_hat - x_prev) / (np.linalg.norm(x_hat) + 1e-12))
        res = y - A @ x_hat
        resn = float(np.linalg.norm(res))

        # ---------- MEMORY: dynamic + peak update ----------
        if measure_memory:
            # dynamic: r, g, Ag, x_tmp, x_hat, res
            dynamic_bytes = (
                r.nbytes
                + g.nbytes
                + Ag.nbytes
                + x_tmp.nbytes
                + x_hat.nbytes
                + res.nbytes
            )
            current_bytes = static_bytes + dynamic_bytes
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes
        # ---------------------------------------------------

        if info is not None and info.iteration_log:
            sparsity = int(np.count_nonzero(x_hat))

            info.add_iteration(
                residual_norm=resn,
                sparsity=sparsity,
                step_size=muy,
                support=None,
            )

        if dx_rel < tol:
            stop_reason = "dx_rel"
            break

        x_prev = x_hat

    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.ISTA_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)

        if measure_memory:
            info.set_meta("memory_static_MB", static_bytes / (1024**2))
            info.set_meta("memory_peak_MB", peak_bytes / (1024**2))

        return x_hat, info

    return x_hat, None
