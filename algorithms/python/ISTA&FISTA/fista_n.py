import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import time


def power_method_lipschitz(
    Phi: np.ndarray, n_iter: int = 50, rng: Optional[np.random.Generator] = None
) -> float:
    m, n = Phi.shape
    if rng is None:
        rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        v = Phi.T @ (Phi @ v)
        nv = np.linalg.norm(v)
        if nv == 0:
            return 1.0
        v /= nv
    L = float(v @ (Phi.T @ (Phi @ v)))
    return max(L, 1e-12)


def fista(
    Phi: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 400,
    tol: float = 1e-6,
    L: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
):

    if L is None:
        L = power_method_lipschitz(Phi, n_iter=30)
    step = 1.0 / L

    n = Phi.shape[1]
    w = np.zeros(n, dtype=Phi.dtype) if x0 is None else x0.astype(Phi.dtype, copy=True)
    z = w.copy()
    tk = 1.0

    hist: Dict[str, list] = {"res": [], "time": []}

    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        # 1) gradient tại z
        r = Phi @ z - y  # matvec #1
        g = Phi.T @ r  # matvec #2

        # 2) bước prox (soft-threshold)
        u = z - step * g
        w_new = np.sign(u) * np.maximum(np.abs(u) - lam * step, 0.0)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        z_new = w_new + ((tk - 1.0) / t_new) * (w_new - w)
        if it == max_iter:
            r_new = Phi @ w_new - y
            hist["res"].append(float(np.linalg.norm(r_new)))
            hist["time"].append(float(time.perf_counter() - t0))

        if np.linalg.norm(w_new - w) <= tol * (np.linalg.norm(w) + 1e-12):
            w = w_new
            break

        w, z, tk = w_new, z_new, t_new

    hist["iters"] = it
    hist["L"] = float(L)
    return w, hist
