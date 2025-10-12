import time
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple


def hard_threshold_topk(w: np.ndarray, k: int) -> np.ndarray:
    if k >= w.size:
        return w
    flat = w.ravel()
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    out = np.zeros_like(flat)
    out[idx] = flat[idx]
    return out.reshape(w.shape)


def IHT(
    y: np.ndarray,
    A: Callable[[np.ndarray], np.ndarray],
    AT: Callable[[np.ndarray], np.ndarray],
    *,
    k: Optional[int] = None,
    thr: Optional[float] = None,
    T_fwd: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    T_inv: Callable[[np.ndarray], np.ndarray] = lambda w: w,
    step: float = 1.0,
    iters: int = 100,
    x_init: Optional[np.ndarray] = None,
    proj: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eps_mode: str = "x",
    eps_abs: float = 1e-4,
    eps_rel: float = 0.0,
    min_iters: int = 5,
    x_ref: Optional[np.ndarray] = None,
):

    if k is None:
        raise ValueError("Pls fill the k")

    x = AT(y) if x_init is None else x_init.copy()

    w = T_fwd(x)

    logs: Dict[str, Any] = {
        "objective": [],
        "residual_norm": [],
        "delta_x": [],
        "mse": [],
        "times": [],
        "stop_reason": None,
        "stop_iter": None,
    }

    t0 = time.time()
    prev_x = x.copy()
    prev_res = None

    for it in range(iters):
        x_t = T_inv(w)
        r_t = y - A(x_t)

        g_t = AT(r_t)

        w = w + step * T_fwd(g_t)

        w = hard_threshold_topk(w, k)

        x = T_inv(w)
        if proj is not None:
            x = proj(x)
        # norm 2 error
        res = y - A(x)
        resn = float(np.linalg.norm(res))

        obj = 0.5 * (resn**2)
        dx = float(np.linalg.norm(x - prev_x))

        logs["residual_norm"].append(resn)
        logs["objective"].append(obj)
        logs["delta_x"].append(dx)
        logs["times"].append(time.time() - t0)

        if x_ref is not None:
            logs["mse"].append(float(np.mean((x - x_ref) ** 2)))
        else:
            logs["mse"].append(np.nan)

        if it + 1 >= min_iters:
            if eps_mode == "x":
                ref = float(np.linalg.norm(x)) + 1e-12
                cond = (dx < eps_abs) or (eps_rel > 0 and dx < eps_rel * ref)
            elif eps_mode == "residual":
                if prev_res is None:
                    cond = False
                else:
                    dres = abs(resn - prev_res)
                    ref = resn + 1e-12
                    cond = (dres < eps_abs) or (eps_rel > 0 and dres < eps_rel * ref)
            else:
                raise ValueError("eps_mode: 'x' or 'residual'.")

            if cond:
                logs["stop_reason"] = f"early_stop_{eps_mode}"
                logs["stop_iter"] = it + 1
                return x, logs

        prev_x = x.copy()
        prev_res = resn

    logs["stop_reason"] = "max_iters"
    logs["stop_iter"] = len(logs["residual_norm"])
    return x, logs
