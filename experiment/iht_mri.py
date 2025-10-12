import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Optional, Tuple
import sys
from skimage.data import shepp_logan_phantom
from skimage.data import camera
from skimage.transform import resize
from scipy.fftpack import dct, idct
import os

save_dir = "results/iht/shepp_phantom"
os.makedirs(save_dir, exist_ok=True)
sys.path.append("../algorithms/python/IHT")
from iht import IHT


def fft2c(img: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) / np.sqrt(img.size)


def ifft2c(ksp: np.ndarray):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp))) * np.sqrt(ksp.size)


def psnr(x: np.ndarray, ref: np.ndarray, data_range: float = 1.0):
    mse = np.mean((x - ref) ** 2)
    if mse <= 1e-18:
        return 99.0
    return 10.0 * np.log10((data_range**2) / mse)


def dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x.T, norm="ortho", type=2).T, norm="ortho", type=2)


def idct2(w: np.ndarray) -> np.ndarray:
    return idct(idct(w.T, norm="ortho", type=2).T, norm="ortho", type=2)


# variable-density sampling
def make_var_dens_mask(ny, nx, accel=6.0, center_fraction=0.1, seed=123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ky = np.linspace(-1, 1, ny)
    kx = np.linspace(-1, 1, nx)
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)
    p = 1 / (1 + (R / 0.3) ** 4)
    p /= p.max()  # normalize
    mask = np.zeros((ny, nx), dtype=bool)
    cy, cx = ny // 2, nx // 2
    wy, wx = int(ny * center_fraction / 2), int(nx * center_fraction / 2)
    mask[cy - wy : cy + wy + 1, cx - wx : cx + wx + 1] = True
    remaining = ~mask
    target = 1.0 / accel
    curr = mask.mean()
    if curr < target:
        scale = (target - curr) / (p[remaining].mean() + 1e-12)
        prob = np.clip(p * scale, 0, 1)
        rand = rng.random((ny, nx))
        add = (rand < prob) & remaining
        mask |= add
    return mask


def make_gaussian_sensing(m: int, n: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Phi = rng.normal(0.0, 1.0, size=(m, n)).astype(np.float32)
    Phi /= np.sqrt(m).astype(np.float32)
    return Phi


def main():
    N = 256
    center_fraction = 0.08
    accel = 4.2
    k_fraction = 0.30
    iters = 200
    step = 1.0
    eps_abs = 1e-5
    min_iters = 20

    # ---- data ----
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), anti_aliasing=True)
    gt = phantom
    mask = make_var_dens_mask(
        N, N, accel=accel, center_fraction=center_fraction, seed=123
    )
    # mask = make_gaussian_sensing(N, N)
    kspace_full = fft2c(gt)
    # y = mask * kspace_full
    # kspace_full = dct2(gt)
    y = mask * kspace_full

    # A = lambda x: mask * dct2(x)
    # AT = lambda yy: idct2(mask * yy)

    A = lambda x: mask * fft2c(x)
    AT = lambda yy: ifft2c(mask * yy)

    zf = np.clip(np.real(AT(y)), 0, 1)
    zf_psnr = psnr(zf, gt)

    # ---- DCT sparsity ----
    T_fwd = lambda x: dct2(x)
    T_inv = lambda w: idct2(w)

    proj = lambda x: np.clip(np.real(x), 0.0, 1.0)

    # ---- IHT run ----
    k = int(k_fraction * gt.size)

    x_rec, logs = IHT(
        y,
        A,
        AT,
        k=k,
        T_fwd=T_fwd,
        T_inv=T_inv,
        step=step,
        iters=iters,
        proj=proj,
        x_init=AT(y),
        eps_mode="x",
        eps_abs=eps_abs,
        eps_rel=0.0,
        min_iters=min_iters,
        x_ref=gt,
    )

    final_psnr = psnr(x_rec, gt)
    runtime = logs["times"][-1]
    stop_reason = logs["stop_reason"]
    stop_iter = logs["stop_iter"]

    print("=== IHT with DCT domain (MRI undersampling) ===")
    print(f"Image size            : {N}x{N}")
    print(f"Accel (~undersampling): {accel}")
    print(f"Center fraction       : {center_fraction}")
    print(f"Sparsity k            : {k}  ({k_fraction*100:.1f}%)")
    print(f"Iterations (max)      : {iters}")
    print(f"Zero-filled  PSNR (dB): {zf_psnr:.2f}")
    print(f"IHT final    PSNR (dB): {final_psnr:.2f}")
    print(f"Runtime (s)           : {runtime:.2f}")
    print(f"Stop reason/iter      : {stop_reason} / {stop_iter}")

    # ---- plots ----
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(x_rec, cmap="gray")
    plt.title(f"IHT (DCT)\nPSNR={final_psnr:.2f} dB")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iht_dct_images.png"), dpi=150)
    plt.show()

    its = np.arange(1, len(logs["objective"]) + 1)
    plt.figure(figsize=(12, 4))
    plt.semilogy(its, logs["objective"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective = 0.5*||y - A x||^2")
    plt.title("Objective vs Iterations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iht_dct_objective.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.semilogy(its, logs["residual_norm"])
    plt.xlabel("Iteration")
    plt.ylabel("||y - A x||_2")
    plt.title("Residual Norm vs Iterations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iht_dct_residual.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.semilogy(its, logs["delta_x"])
    plt.xlabel("Iteration")
    plt.ylabel("||x_t - x_{t-1}||_2")
    plt.title("Δx vs Iterations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "iht_dct_delta_x.png"), dpi=150)
    plt.show()

    print(f"\n✅ Kết quả và hình ảnh đã lưu tại: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
