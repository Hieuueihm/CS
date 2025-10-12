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


def ista(
    Phi: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 600,
    tol: float = 1e-6,
    L: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
    stop_on_residual: bool = False,
    res_tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict]:

    m, n = Phi.shape
    if L is None:
        L = power_method_lipschitz(Phi, n_iter=30)
    step = 1.0 / L

    w = np.zeros(n, dtype=Phi.dtype) if x0 is None else x0.astype(Phi.dtype, copy=True)
    hist = {"res": [], "time": []}
    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        # r = Phi w - y
        r = Phi @ w
        r -= y

        # g = Phi^T r
        g = Phi.T @ r

        # z = w - step*g ; soft-threshold
        z = w - step * g
        # w_new = S_{lam*step}(z)
        th = lam * step

        absz = np.abs(z)
        w_new = np.sign(z) * np.maximum(absz - th, 0.0)

        if np.linalg.norm(w_new - w) <= tol * (np.linalg.norm(w) + 1e-12):
            w = w_new
        if stop_on_residual and (np.linalg.norm(r) <= res_tol):
            w = w_new

        w = w_new

    hist["iters"] = it
    hist["L"] = float(L)
    return w, hist
