

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.fft import dctn, idctn
from skimage import data, img_as_float32
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.io import imread, imsave

# =====================================================
# 1) CÔNG CỤ CHUNG (không class)
# =====================================================

def vectorize(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)


def devectorize(v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return v.reshape(shape)


def psnr(x: np.ndarray, x_ref: np.ndarray, data_range: float = 1.0) -> float:
    return float(sk_psnr(x_ref, x, data_range=data_range))


# =====================================================
# 2) ẢNH MẪU
# =====================================================
SK_BUILTINS = {
    "camera": data.camera,
    "astronaut": data.astronaut,
    "coffee": data.coffee,
    "chelsea": data.chelsea,
    "rocket": data.rocket,
    "page": data.page,
    "coins": data.coins,
}

def load_image(src: str = "camera", size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if src in SK_BUILTINS:
        img = SK_BUILTINS[src]()
    else:
        img = imread(src)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img_as_float32(img)
    if size is not None:
        H, W = size
        h, w = img.shape
        ys = (np.linspace(0, h - 1, H)).astype(np.int32)
        xs = (np.linspace(0, w - 1, W)).astype(np.int32)
        img = img[ys][:, xs]
    return img


# =====================================================
# 3) MIỀN CƠ SỞ DẠNG HÀM (T_fwd, T_inv)
# =====================================================

def make_basis(name: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    name = name.lower()
    if name == "dct":
        fwd = lambda x: dctn(x, type=2, norm="ortho")
        inv = lambda c: idctn(c, type=2, norm="ortho")
        return (fwd, inv, "DCT2(ortho)")
    elif name == "fft":
        fwd = lambda x: fft2(x)
        inv = lambda c: np.real(ifft2(c))
        return (fwd, inv, "FFT2")
    else:
        raise ValueError(f"Unknown basis: {name}")


# =====================================================
# 4) TOÁN TỬ ĐO DẠNG HÀM (A, AT)
# =====================================================

def build_gaussian_ops(img_shape: Tuple[int, int], m: int, seed: Optional[int] = 0) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    H, W = img_shape
    n = H * W
    rng = np.random.default_rng(seed)
    A_mat = rng.standard_normal((m, n)).astype(np.float32) / math.sqrt(m)
    A = lambda v: A_mat @ v
    AT = lambda v: A_mat.T @ v
    name = f"GaussianA(m={m}, n={n})"
    return A, AT, name


def build_partial_fourier_ops(img_shape: Tuple[int, int], samp_frac: float, seed: Optional[int] = 0) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    H, W = img_shape
    n = H * W
    m = int(round(samp_frac * n))
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=bool)
    idx = rng.choice(H * W, size=m, replace=False)
    mask.flat[idx] = True

    def A(v: np.ndarray) -> np.ndarray:
        x = devectorize(v, (H, W))
        Xf = fft2(x)
        return Xf[mask].view(np.complex64)

    def AT(y: np.ndarray) -> np.ndarray:
        Yfull = np.zeros((H, W), dtype=np.complex64)
        Yfull[mask] = y
        x_rec = ifft2(Yfull)
        return np.real(x_rec).reshape(-1)

    name = f"PartialFourier(m={m}, n={n})"
    return A, AT, name


def make_operator(kind: str, img_shape: Tuple[int, int], m_ratio: float, seed: int) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    n = img_shape[0] * img_shape[1]
    m = int(round(m_ratio * n))
    kind = kind.lower()
    if kind == "gaussian":
        return build_gaussian_ops(img_shape, m, seed)
    elif kind in ("pfft", "partial_fourier", "fourier"):
        return build_partial_fourier_ops(img_shape, samp_frac=m_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown measurement kind: {kind}")


# =====================================================
# 5) NHIỄU
# =====================================================

def add_noise(y: np.ndarray, *, sigma: Optional[float] = None, snr_db: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if np.iscomplexobj(y):
        if snr_db is not None:
            p_signal = np.mean(np.abs(y) ** 2)
            sigma2 = p_signal / (10 ** (snr_db / 10))
            sigma = math.sqrt(sigma2 / 2)
        else:
            sigma = sigma or 0.0
        noise = rng.normal(0, sigma, size=y.shape) + 1j * rng.normal(0, sigma, size=y.shape)
        return y + noise
    else:
        if snr_db is not None:
            p_signal = np.mean(y ** 2)
            sigma2 = p_signal / (10 ** (snr_db / 10))
            sigma = math.sqrt(sigma2)
        else:
            sigma = sigma or 0.0
        noise = rng.normal(0, sigma, size=y.shape)
        return y + noise


# =====================================================
# 6) THUẬT TOÁN IHT (NGUYÊN BẢN BẠN CUNG CẤP)
# =====================================================

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
    if (k is None):
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
        res = y - A(x)
        resn = float(np.linalg.norm(res))
        obj  = 0.5 * (resn ** 2)
        dx   = float(np.linalg.norm(x - prev_x))

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
                raise ValueError("eps_mode phải là 'x' hoặc 'residual'.")
            if cond:
                logs["stop_reason"] = f"early_stop_{eps_mode}"
                logs["stop_iter"] = it + 1
                return x, logs

        prev_x = x.copy()
        prev_res = resn

    logs["stop_reason"] = "max_iters"
    logs["stop_iter"] = len(logs["residual_norm"])
    return x, logs


# =====================================================
# 7) LOGGER LƯU KẾT QUẢ (functional)
# =====================================================

def _exp_dir(root: str, cfg: Dict[str, Any]) -> Path:
    Path(root).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"img={cfg['image']}_basis={cfg['basis']}_meas={cfg['meas_kind']}_mr={cfg['m_ratio']:.2f}_alg={cfg['algorithm_name']}"
    d = Path(root) / f"{ts}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)
    return d


def save_image(path: Path, img: np.ndarray):
    x = np.clip(img, 0, 1)
    imsave(str(path), (x * 255).astype(np.uint8))


def save_json(path: Path, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =====================================================
# 8) PIPELINE THỰC NGHIỆM (functional, gọi thuật toán là HÀM)
# =====================================================

def run_experiment(cfg: Dict[str, Any], out_root: str = "results") -> Path:
    print("run_experiment")
    # 1) ảnh
    x0 = load_image(cfg.get("image", "camera"), cfg.get("image_size"))
    H, W = x0.shape

    # 2) basis → (T_fwd_img, T_inv_img)
    T_fwd_img, T_inv_img, basis_name = make_basis(cfg.get("basis", "dct"))

    # 3) operator → (A, AT)
    A, AT, op_name = make_operator(cfg.get("meas_kind", "gaussian"), (H, W), cfg.get("m_ratio", 0.25), cfg.get("seed", 0))

    # 4) measurements
    vec_x = vectorize(x0)
    y_clean = A(vec_x)
    noise = cfg.get("noise", {"snr_db": 40.0})
    y_noisy = add_noise(y_clean, sigma=noise.get("sigma"), snr_db=noise.get("snr_db"))

    # 5) thuật toán phục hồi — HÀM (ví dụ: IHT)
    alg_fn: Callable = cfg["algorithm_fn"]
    alg_kwargs: Dict[str, Any] = cfg.get("algorithm_kwargs", {})

    # T_fwd/T_inv của thuật toán hoạt động trên VECTOR, nên wrap:
    T_fwd = lambda x_vec: vectorize(T_fwd_img(devectorize(x_vec, (H, W))))
    T_inv = lambda w_vec: vectorize(T_inv_img(devectorize(w_vec, (H, W))))

    # x_init nếu muốn: None → thuật toán sẽ dùng AT(y) làm khởi tạo
    x_init = cfg.get("x_init", None)
    x_ref_vec = vectorize(x0) if cfg.get("log_psnr", True) else None

    t0 = time.time()
    x_vec, logs = alg_fn(
        y=y_noisy,
        A=A,
        AT=AT,
        T_fwd=T_fwd,
        T_inv=T_inv,
        x_init=x_init,
        x_ref=x_ref_vec,
        iters=cfg.get("max_iter", 50),
        **alg_kwargs,
    )
    elapsed = time.time() - t0

    x_hat = devectorize(x_vec, (H, W))
    out_psnr = psnr(x_hat, x0, data_range=1.0)

    # 6) lưu
    logcfg = dict(cfg)
    logcfg.update({
        "algorithm_name": cfg.get("algorithm_name", alg_fn.__name__),
        "basis_name": basis_name,
        "operator": op_name,
    })
    out_dir = _exp_dir(out_root, logcfg)

    save_image(out_dir / "gt.png", x0)
    # nếu là Partial Fourier, lưu zero-filled tham khảo
    if "PartialFourier" in op_name:
        zf_img = devectorize(AT(y_noisy), (H, W))
        save_image(out_dir / "zerofilled.png", zf_img)
    save_image(out_dir / "recon.png", x_hat)

    # số liệu
    np.save(out_dir / "y_clean.npy", y_clean)
    np.save(out_dir / "y_noisy.npy", y_noisy)
    np.save(out_dir / "x_gt.npy", x0)
    np.save(out_dir / "x_hat.npy", x_hat)
    for k, v in logs.items():
        try:
            np.save(out_dir / f"hist_{k}.npy", np.array(v))
        except Exception:
            pass

    save_json(out_dir / "metrics.json", {
        "operator": op_name,
        "basis": basis_name,
        "elapsed_sec": elapsed,
        "psnr": out_psnr,
        "history_keys": list(logs.keys()),
    })

    return out_dir


# =====================================================
# 9) CLI DEMO — chạy IHT **dạng hàm** trên danh sách ảnh có sẵn
# =====================================================
if __name__ == "__main__":
    image_list = ["camera"]
    for img in image_list:
        cfg = {
            "image": img,
            "image_size": (256, 256),
            "basis": "dct",               # hoặc 'fft'
            "meas_kind": "gaussian",       # hoặc 'pfft'
            "m_ratio": 0.25,
            "seed": 42,
            "noise": {"snr_db": 40.0},
            "algorithm_fn": IHT,           # <<< dùng trực tiếp HÀM IHT của bạn
            "algorithm_name": "IHT",
            "algorithm_kwargs": {          # tham số riêng của IHT
                "k": int(0.05 * 256 * 256),   # 5% hệ số
                "step": 1.0,
                "eps_mode": "x",
                "eps_abs": 1e-4,
                "eps_rel": 0.0,
                "min_iters": 5,
            },
            "max_iter": 50,
            "log_psnr": True,
        }
        out = run_experiment(cfg)
        print(f"[IHT] {img} -> saved to: {out}")
