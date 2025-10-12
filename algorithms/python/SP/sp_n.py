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


def sp(
    Phi: np.ndarray,
    y: np.ndarray,
    K: int,
    max_iter: int = 50,
    tol: float = 1e-10,
    rels: bool = True,
) -> Tuple[np.ndarray, Dict]:
    m, n = Phi.shape
    u0 = Phi.T @ y
    S = topk_indices_abs(u0, K).tolist()
    AS = Phi[:, S]
    xS, *_ = np.linalg.lstsq(AS, y, rcond=None)
    w = np.zeros(n)
    w[S] = xS
    r = y - AS @ xS
    hist = {"res": [], "time": []}
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        u = Phi.T @ r
        T = topk_indices_abs(u, K).tolist()
        U = list(set(S).union(T))
        AU = Phi[:, U]
        bU, *_ = np.linalg.lstsq(AU, y, rcond=None)
        b = np.zeros(n)
        b[U] = bU
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
        if (set(S_new) == set(S)) or (np.linalg.norm(r) - np.linalg.norm(r_new) <= tol):
            w = w_new
            break
        w = w_new
        r = r_new
        S = S_new
    hist["iters"] = it
    return w, hist
