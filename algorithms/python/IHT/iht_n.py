import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import time


def power_method_lipschitz(
    Phi: np.ndarray, n_iter: int = 50, rng: Optional[np.random.Generator] = None
):
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


def topk_indices_abs(v: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, v.size))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(np.abs(v), -k)[-k:]
    idx = idx[np.argsort(-np.abs(v[idx]))]
    return idx


def iht(
    Phi: np.ndarray,
    y: np.ndarray,
    K: int,
    mu: Optional[float] = None,
    max_iter: int = 300,
    tol: float = 1e-6,
    x0: Optional[np.ndarray] = None,
):
    m, n = Phi.shape
    if mu is None:
        L = power_method_lipschitz(Phi, n_iter=30)
        mu = 1.0 / L
    w = np.zeros(n) if x0 is None else x0.copy()
    hist = {"res": [], "time": []}
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        r = y - Phi @ w
        g = Phi.T @ r
        z = w + mu * g
        idx = topk_indices_abs(z, K)
        w_new = np.zeros_like(w)
        w_new[idx] = z[idx]
        hist["res"].append(float(np.linalg.norm(y - Phi @ w_new)))
        hist["time"].append(float(time.perf_counter() - t0))
        if np.linalg.norm(w_new - w) <= tol * (np.linalg.norm(w) + 1e-12):
            w = w_new
            break
        w = w_new
    hist["iters"] = it
    hist["mu"] = mu
    return w, hist
