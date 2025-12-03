import numpy as np
import time
from core.metrics import psnr


def cs_iht(
    y,
    A,
    k=None,
    max_iter=1000,
    tol=1e-6,
    step=0.5,
    return_info=True,
    ignore_iteration_log=False,
    measure_memory=False,
    x_true=None,
    psnr_threshold=None,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    Iterative Hard Thresholding (IHT) for Compressive Sensing
    y ≈ A x, A is (m, n)

    Parameters
    ----------
    y : ndarray, shape (m,)
        Measurements.
    A : ndarray, shape (m, n)
        Sensing / measurement matrix.
    k : int, optional
        Sparsity level. If None, default = m // 4.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Relative change tolerance on residual norm.
    step : float, optional
        Gradient step size (u). If not used, will be overwritten by 1/L.
    return_info : bool, optional
        Whether to return AlgorithmInformation.
    ignore_iteration_log : bool, optional
        If True, do not log per-iteration information.
    measure_memory : bool, optional
        If True, estimate static and peak memory usage based on array sizes.

    Returns
    -------
    hat_x : ndarray, shape (n,)
        Recovered sparse signal.
    info : AlgorithmInformation (optional)
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, n = A.shape

    if k is None:
        k = m // 4

    info = AlgorithmInformation(algorithm_name="iht") if return_info else None

    hat_x_prev = np.zeros(n)

    prev_resn = None
    t0 = time.time()
    stop_reason = "max_iter_reached"

    # Lipschitz constant L = ||A||_2^2
    L = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / L

    # ---------- MEMORY: static + peak (ước lượng) ----------
    if measure_memory:
        # static: A, y, hat_x_prev
        static_bytes = A.nbytes + y.nbytes + hat_x_prev.nbytes
        peak_bytes = static_bytes
    else:
        static_bytes = None
        peak_bytes = None
    # -------------------------------------------------------

    if info is not None:
        if ignore_iteration_log:
            info.iteration_log = False
        info.set_meta("k", k)
        info.set_meta("max_iter", max_iter)
        info.set_meta("step", step)

    for _ in range(1, max_iter + 1):
        # Gradient step
        grad = A.T @ (y - A @ hat_x_prev)
        x_tmp = hat_x_prev + step * grad

        flat = x_tmp.ravel()
        if k < flat.size:
            idx_topk = np.argpartition(np.abs(flat), -k)[-k:]
            x_ht = np.zeros_like(flat)
            x_ht[idx_topk] = flat[idx_topk]
            hat_x = x_ht.reshape(x_tmp.shape)
        else:
            hat_x = x_tmp

        residual = y - A @ hat_x
        resn = float(np.linalg.norm(residual))

        # ---------- MEMORY: dynamic + peak update ----------
        if measure_memory:
            # dynamic: grad, hat_x, residual
            dynamic_bytes = grad.nbytes + hat_x.nbytes + residual.nbytes
            current_bytes = static_bytes + dynamic_bytes
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes
        # ---------------------------------------------------

        if info is not None and info.iteration_log:
            sparsity = int(np.count_nonzero(hat_x))
            info.add_iteration(
                residual_norm=resn,
                sparsity=sparsity,
                step_size=step,
            )

        if prev_resn is not None:
            rel_change = abs(prev_resn - resn) / max(prev_resn, 1e-12)
            if rel_change <= tol:
                stop_reason = "residual_converged"
                break
        if resn <= tol:
            stop_reason = "residual_below_tol"
            break
        if x_true is not None:
            current_psnr = psnr(x_true, hat_x, data_range=1.0)
            if current_psnr >= psnr_threshold:
                stop_reason = "psnr_threshold_reached"
                break

        prev_resn = resn
        hat_x_prev = hat_x

    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.IHT_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)

        if measure_memory:
            info.set_meta("memory_static_MB", static_bytes / (1024**2))
            info.set_meta("memory_peak_MB", peak_bytes / (1024**2))

        return hat_x, info

    return hat_x, None
