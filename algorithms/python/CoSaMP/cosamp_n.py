import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import time


def topk_indices_abs(v: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, v.size))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(np.abs(v), -k)[-k:]
    idx = idx[np.argsort(-np.abs(v[idx]))]
    return idx


def cosamp(
    Phi: np.ndarray,
    y: np.ndarray,
    K: int,
    max_iter: int = 50,
    tol: float = 1e-10,
    rels: bool = False,
) -> Tuple[np.ndarray, Dict]:
    m, n = Phi.shape
    r = y.copy()
    S: List[int] = []
    w = np.zeros(n)
    hist = {"res": [], "time": []}
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        u = Phi.T @ r
        Omega = topk_indices_abs(u, 2 * K).tolist()
        T = list(set(S).union(Omega))  # <= 3K
        AT = Phi[:, T]
        bT, *_ = np.linalg.lstsq(AT, y, rcond=None)
        b = np.zeros(n)
        b[T] = bT
        S_new = topk_indices_abs(b, K).tolist()
        if rels:
            AS = Phi[:, S_new]
            xS, *_ = np.linalg.lstsq(AS, y, rcond=None)
            w_new = np.zeros(n)
            w_new[S_new] = xS
        else:
            w_new = np.zeros(n)
            w_new[S_new] = b[S_new]
        r_new = y - Phi @ w_new
        hist["res"].append(float(np.linalg.norm(r_new)))
        hist["time"].append(float(time.perf_counter() - t0))
        if np.linalg.norm(r) - np.linalg.norm(r_new) <= tol:
            w = w_new
            break
        w = w_new
        r = r_new
        S = S_new
    hist["iters"] = it
    return w, hist
