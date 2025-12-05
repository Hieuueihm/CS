import numpy as np
import time
from core.metrics import psnr


def cs_fista(
    y,
    A,
    lambda_=2e-5,
    tol=1e-4,
    max_iter=10000,
    return_info=True,
    ignore_iteration_log=False,
    measure_memory=False,
):
    from utils.algorithm_information import AlgorithmInformation

    """
    FISTA (Fast ISTA) for L1-regularized Compressive Sensing

    Solve approximately:
        min_x  0.5 * ||y - A x||_2^2 + lambda_ * ||x||_1
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

    # Khởi tạo
    x_prev = np.zeros(n)
    x_curr = np.zeros(n)
    z_curr = x_curr.copy()  # điểm có momentum (y_k trong paper)
    t_curr = 1.0

    # L = ||A||_2^2, bước μ = 1/L
    L = np.linalg.norm(A, 2) ** 2
    muy = 1.0 / (L + 1e-12)

    # --------- MEMORY: static + peak (ước lượng) ----------
    if measure_memory:
        # static: A, y, x_prev, x_curr, z_curr
        static_bytes = (
            A.nbytes + y.nbytes + x_prev.nbytes + x_curr.nbytes + z_curr.nbytes
        )
        peak_bytes = static_bytes
    else:
        static_bytes = None
        peak_bytes = None
    # ------------------------------------------------------

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for it in range(1, max_iter + 1):
        # 1) Gradient tại z_curr (y_k)
        r_z = y - A @ z_curr  # residual
        g_z = A.T @ r_z  # = -∇f(z_curr)
        u = z_curr + muy * g_z  # z_curr - (1/L)∇f(z_curr)

        # 2) Soft-thresholding
        x_next = np.sign(u) * np.maximum(np.abs(u) - muy * lambda_, 0.0)

        # 3) Kiểm tra hội tụ (trước khi cập nhật momentum)
        dx_rel = float(
            np.linalg.norm(x_next - x_curr) / (np.linalg.norm(x_next) + 1e-12)
        )
        res = y - A @ x_next
        resn = float(np.linalg.norm(res))

        # ---------- MEMORY: dynamic + peak update ----------
        if measure_memory:
            # dynamic: r_z, g_z, u, x_next, res
            dynamic_bytes = (
                r_z.nbytes + g_z.nbytes + u.nbytes + x_next.nbytes + res.nbytes
            )
            current_bytes = static_bytes + dynamic_bytes
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes
        # ---------------------------------------------------

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
            x_curr = x_next
            break

        # 4) Cập nhật momentum SAU khi có x_next
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_curr**2)) / 2.0
        z_next = x_next + ((t_curr - 1.0) / t_next) * (x_next - x_curr)

        # 5) Chuẩn bị cho vòng lặp sau
        x_prev = x_curr
        x_curr = x_next
        z_curr = z_next
        t_curr = t_next

    x_hat = x_curr
    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.FISTA_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)

        if measure_memory:
            info.set_meta("memory_static_MB", static_bytes / (1024**2))
            info.set_meta("memory_peak_MB", peak_bytes / (1024**2))

        return x_hat, info

    return x_hat, None
