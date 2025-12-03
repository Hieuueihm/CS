import numpy as np
import time
from core.metrics import psnr


def cs_sp(
    y,
    A,
    k=None,
    max_iter=50,
    tol=1e-4,
    return_info=True,
    ignore_iteration_log=False,
    measure_memory=False,
    x_true=None,
    psnr_threshold=None,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    Subspace Pursuit (SP) for Compressive Sensing

    Parameters
    ----------
    y : ndarray, shape (m,)
        Measurement vector.
    A : ndarray, shape (m, n)
        Sensing / measurement matrix.
    k : int, optional
        Sparsity level. If None, default = m // 4.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Residual tolerance for stopping.
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
    """

    info = AlgorithmInformation(algorithm_name="sp") if return_info else None
    if info is not None and ignore_iteration_log:
        info.iteration_log = False

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    m, n = A.shape

    if k is None:
        k = m // 4  # 25% of m

    if info is not None:
        info.set_meta("max_iter", max_iter)
        info.set_meta("tol", tol)
        info.set_meta("k", k)

    hat_x = np.zeros(n)
    residual = y.copy()  # r_0 = y
    support_prev = np.array([], dtype=int)

    # --------- MEMORY: static + peak (ước lượng) ----------
    if measure_memory:
        # static: A, y, hat_x, residual ban đầu
        static_bytes = A.nbytes + y.nbytes + hat_x.nbytes + residual.nbytes
        peak_bytes = static_bytes
    else:
        static_bytes = None
        peak_bytes = None
    # ------------------------------------------------------

    t0 = time.time()
    stop_reason = "max_iter_reached"
    prev_resn = None

    for _ in range(1, max_iter + 1):
        # Correlation: proxy = A^T r
        proxy = A.T @ residual  # (n,)

        # indices của k phần tử lớn nhất trong proxy
        idx_sorted = np.argsort(np.abs(proxy))
        omega = idx_sorted[-k:]

        # candidate support = union(omega, support_prev)
        candidate_support = np.union1d(omega, support_prev).astype(int)

        # Least-squares trên candidate_support
        A_S = A[:, candidate_support]  # (m, |S|)
        b_S, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)  # (|S|,)

        # chọn lại top-k từ b_S
        idx_sorted_b = np.argsort(np.abs(b_S))
        idx_k = idx_sorted_b[-k:]
        new_support = candidate_support[idx_k]

        x_temp = np.zeros(n)
        x_temp[new_support] = b_S[idx_k]

        # residual mới
        residual = y - A @ x_temp
        r_norm = np.linalg.norm(residual)

        # ---------- MEMORY: dynamic + peak update ----------
        if measure_memory:
            # dynamic: proxy, A_S, b_S, x_temp, residual
            dynamic_bytes = (
                proxy.nbytes + A_S.nbytes + b_S.nbytes + x_temp.nbytes + residual.nbytes
            )
            current_bytes = static_bytes + dynamic_bytes
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes
        # ---------------------------------------------------
        if info is not None and info.iteration_log:
            info.add_iteration(
                residual_norm=r_norm,
                sparsity=int(np.count_nonzero(x_temp)),
                support=new_support.copy(),
            )
        if prev_resn is not None:
            rel_change = abs(prev_resn - r_norm) / max(prev_resn, 1e-12)
            if rel_change <= tol:
                stop_reason = "residual_converged"
                break
        if r_norm < tol:
            stop_reason = "tol_reached"
            break
            # Update support
        if x_true is not None:
            current_psnr = psnr(x_true, hat_x, data_range=1.0)
            if current_psnr >= psnr_threshold:
                stop_reason = "psnr_threshold_reached"
                break
        support_prev = new_support
        prev_resn = r_norm

    t1 = time.time()
    hat_x = x_temp

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.SP_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)

        if measure_memory:
            info.set_meta("memory_static_MB", static_bytes / (1024**2))
            info.set_meta("memory_peak_MB", peak_bytes / (1024**2))

        return hat_x, info

    return hat_x, None
