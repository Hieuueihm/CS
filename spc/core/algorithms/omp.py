import numpy as np
import time
from core.metrics import psnr


def cs_omp(
    y,
    A,
    k=None,
    max_iter=1000,
    tol=1e-6,
    return_info=True,
    ignore_iteration_log=False,
    measure_memory=False,
):
    from utils.algorithm_information import AlgorithmInformation

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    info = AlgorithmInformation(algorithm_name="omp") if return_info else None
    if info is not None and ignore_iteration_log:
        print("Ignoring iteration log as requested.")
        info.iteration_log = False

    m, n = A.shape

    hat_x = np.zeros(n)
    residual = y.copy()
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

    if measure_memory:
        static_bytes = A.nbytes + y.nbytes + hat_x.nbytes + residual.nbytes
        peak_bytes = static_bytes
    else:
        static_bytes = None
        peak_bytes = None

    for it in range(n_iter):
        correlations = A.T @ residual

        if measure_memory:
            if it == 0:
                static_bytes += correlations.nbytes

        pos = int(np.argmax(np.abs(correlations)))

        if pos not in support:
            support.append(pos)

        A_S = A[:, support]

        x_S, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)

        residual = y - A_S @ x_S

        if info is not None and info.iteration_log:
            info.add_iteration(
                residual_norm=np.linalg.norm(residual),
                sparsity=len(support),
                support=support.copy(),
            )

        if measure_memory:
            dynamic_bytes = A_S.nbytes
            if x_S is not None:
                dynamic_bytes += x_S.nbytes

            current_bytes = static_bytes + dynamic_bytes
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes

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

            if measure_memory:
                info.set_meta("memory_static_MB", static_bytes / (1024**2))
                info.set_meta("memory_peak_MB", peak_bytes / (1024**2))

        print("OMP finished:", stop_reason)
        return hat_x, info

    return hat_x, None
