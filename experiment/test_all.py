# cs_dct_recovery_full.py
import time, os, json
import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
from skimage.data import camera
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
import sys

sys.path.append("../algorithms/python/ISTA&FISTA")
sys.path.append("../algorithms/python/OMP")
sys.path.append("../algorithms/python/CoSaMP")
sys.path.append("../algorithms/python/SP")
sys.path.append("../algorithms/python/IHT")
from ista_n import ista
from fista_n import fista
from omp_n import omp
from cosamp_n import cosamp
from sp_n import sp
from iht_n import iht


def dct2_orth(x: np.ndarray) -> np.ndarray:
    return dctn(x, type=2, norm="ortho")


def idct2_orth(X: np.ndarray) -> np.ndarray:
    return idctn(X, type=2, norm="ortho")


def psnr(x: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((x - ref) ** 2)
    if mse <= 1e-18:
        return 99.0
    return 10.0 * np.log10((data_range**2) / mse)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, header: List[str], rows: List[List[float]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def make_measurement_matrix(
    m: int, n: int, kind: str, rng: np.random.Generator
) -> np.ndarray:

    if kind == "gaussian":
        Phi = rng.standard_normal((m, n), dtype=np.float64) / np.sqrt(m)
    elif kind == "bernoulli":
        Phi = (rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1).astype(
            np.float64
        ) / np.sqrt(m)
    else:
        raise ValueError("kind must be 'gaussian' or 'bernoulli'")
    return Phi


def recon_from_w(w_vec: np.ndarray, N: int) -> np.ndarray:
    W = w_vec.reshape(N, N)
    x_rec = idct2_orth(W)
    return np.clip(x_rec, 0.0, 1.0)


def run_one_case(
    algo_name: str,
    solver: Callable,
    Phi: np.ndarray,
    y: np.ndarray,
    N: int,
    x_gt_img: np.ndarray,
    params: Dict,
    out_dir: str,
):
    t0 = time.perf_counter()
    w_hat, hist = solver(Phi, y, **params)
    elapsed = time.perf_counter() - t0
    x_rec = recon_from_w(w_hat, N)
    rec_psnr = psnr(x_rec, x_gt_img)

    ensure_dir(out_dir)
    from matplotlib import cm

    plt.imsave(os.path.join(out_dir, "recon.png"), x_rec, cmap=cm.gray)
    plt.imsave(os.path.join(out_dir, "gt.png"), x_gt_img, cmap=cm.gray)

    rows = []

    for i, (tsec, res) in enumerate(zip(hist["time"], hist["res"]), start=1):
        rows.append([i, tsec, res])
    save_csv(
        os.path.join(out_dir, "convergence.csv"),
        ["iter", "time_sec", "res_norm"],
        rows,
    )

    try:
        fig = plt.figure()
        plt.plot([r[0] for r in rows], [r[2] for r in rows])
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.title(f"{algo_name} convergence")
        fig.savefig(
            os.path.join(out_dir, "convergence_residual.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception:
        pass

    # metadata
    meta = {
        "algo": algo_name,
        "elapsed_sec": elapsed,
        "iters": hist.get("iters", None),
        "metrics": {"PSNR": float(rec_psnr)},
        "solver_params": params,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "psnr": rec_psnr,
        "time": elapsed,
        "iters": hist.get("iters", None),
    }


def main():
    N = 64
    m_ratio = 0.25
    measure_kinds = ["gaussian", "bernoulli"]
    noise_rel_list = [0.0, 0.01]
    seed = 0

    img = camera().astype(np.float64) / 255.0
    if img.ndim == 3:
        img = img[..., 0]
    if img.shape[0] != N or img.shape[1] != N:
        img = resize(img, (N, N), anti_aliasing=True)
    x_gt = img
    W_gt = dct2_orth(x_gt)
    w_gt = W_gt.reshape(-1)
    n = w_gt.size

    ensure_dir("results")
    summary_rows = []
    summary_header = [
        "algo",
        "measure",
        "noise_rel",
        "PSNR",
        "SSIM",
        "time_sec",
        "iters",
        "m",
        "n",
        "K",
        "lambda",
        "seed",
    ]

    for measure_kind in measure_kinds:
        rng = np.random.default_rng(seed)
        m = int(round(m_ratio * n))
        Phi = make_measurement_matrix(m, n, measure_kind, rng)

        y_clean = Phi @ w_gt
        y_std = float(np.std(y_clean)) + 1e-12

        PhiTy = Phi.T @ y_clean
        lam = 1e-3 * np.linalg.norm(PhiTy, ord=np.inf)
        K = max(1, m // 4)

        for noise_rel in noise_rel_list:
            noise_sigma = noise_rel * y_std
            rng = np.random.default_rng(seed)  #
            noise = noise_sigma * rng.standard_normal(m)
            y = y_clean + noise

            case_tag = f"{measure_kind}/noise_{noise_rel:.3f}"
            jobs = [
                ("ISTA", ista, dict(lam=lam, max_iter=600, tol=1e-6)),
                ("FISTA", fista, dict(lam=lam, max_iter=600, tol=1e-6)),
                ("IHT", iht, dict(K=K, max_iter=600, tol=1e-6)),
                ("OMP", omp, dict(K=K, tol=1e-6)),
                ("SP", sp, dict(K=K, max_iter=600, tol=1e-6, rels=True)),
                ("CoSaMP", cosamp, dict(K=K, max_iter=600, tol=1e-6, rels=False)),
            ]

            for algo_name, solver, params in jobs:
                out_dir = os.path.join("results", algo_name, case_tag)
                stats = run_one_case(
                    algo_name, solver, Phi, y, N, x_gt, params, out_dir
                )
                summary_rows.append(
                    [
                        algo_name,
                        measure_kind,
                        noise_rel,
                        f"{stats['psnr']:.4f}",
                        f"{stats['time']:.4f}",
                        stats.get("iters", None),
                        m,
                        n,
                        K,
                        f"{lam:.3e}",
                        seed,
                    ]
                )
                print(
                    f"[{algo_name:6s}] measure={measure_kind:9s}, noise_rel={noise_rel:.3f} "
                    f"-> PSNR={stats['psnr']:.2f} dB, time={stats['time']:.3f}s"
                )

    save_csv(os.path.join("results", "summary.csv"), summary_header, summary_rows)
    print("\nSaved logs & images under ./results/<ALGO>/<measure>/<noise_...>/")
    print("Overall summary: results/summary.csv")


if __name__ == "__main__":
    main()
