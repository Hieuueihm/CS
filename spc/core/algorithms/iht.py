import numpy as np
import time


def cs_iht(
    y,
    A,
    k=None,
    max_iter=1000,
    patience=10,
    step=0.5,
    return_info=True,
    ignore_iteration_log=False,
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
    patience : int, optional
        Number of consecutive iterations with unchanged support
        before stopping (convergence criterion).
    step : float, optional
        Gradient step size (u).
    return_info : bool, optional
        Whether to return AlgorithmInformation.
    ignore_iteration_log : bool, optional
        If True, do not log per-iteration information.

    Returns
    -------
    hat_x : ndarray, shape (n,)
        Recovered sparse signal.
    """

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, n = A.shape

    if k is None:
        k = m // 4

    info = AlgorithmInformation(algorithm_name="iht") if return_info else None
    if info is not None:
        if ignore_iteration_log:
            info.iteration_log = False
        info.set_meta("k", k)
        info.set_meta("max_iter", max_iter)
        info.set_meta("patience", patience)
        info.set_meta("step", step)

    hat_x_prev = np.zeros(n)
    last_support = set()
    stable_count = 0

    t0 = time.time()
    stop_reason = "max_iter_reached"

    for _ in range(1, max_iter + 1):
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

        support = set(np.nonzero(hat_x)[0])

        residual = y - A @ hat_x
        resn = float(np.linalg.norm(residual))

        if info is not None and info.iteration_log:
            sparsity = int(np.count_nonzero(hat_x))

            info.add_iteration(
                residual_norm=resn,
                sparsity=sparsity,
                step_size=step,
                support=sorted(support),
            )

        if support == last_support:
            stable_count += 1
        else:
            stable_count = 0
        last_support = support

        if stable_count >= patience:
            stop_reason = "support_stable"
            break

        hat_x_prev = hat_x

    t1 = time.time()

    if return_info and info is not None:
        info.set_stop_reason(
            AlgorithmInformation.IHT_STOP_INFORMATION + ":" + stop_reason
        )
        info.set_time(t1 - t0)
        return hat_x, info

    return hat_x
