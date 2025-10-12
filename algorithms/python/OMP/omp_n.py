import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import time


def omp(Phi: np.ndarray, y: np.ndarray, K: int, tol: float = 1e-10):
    m, n = Phi.shape
    r = y.copy()
    S: List[int] = []
    w = np.zeros(n)
    hist = {"res": [], "time": []}
    t0 = time.perf_counter()
    for t in range(1, K + 1):
        corr = Phi.T @ r
        i = int(np.argmax(np.abs(corr)))
        if i not in S:
            S.append(i)
        AS = Phi[:, S]
        coeff, *_ = np.linalg.lstsq(AS, y, rcond=None)
        w.fill(0.0)
        w[np.array(S)] = coeff
        r = y - AS @ coeff
        hist["res"].append(float(np.linalg.norm(r)))
        hist["time"].append(float(time.perf_counter() - t0))
        if np.linalg.norm(r) <= tol:
            break
    hist["iters"] = len(S)
    return w, hist
