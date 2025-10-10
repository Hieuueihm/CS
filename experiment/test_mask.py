import numpy as np
import matplotlib.pyplot as plt

def make_var_dens_mask(ny, nx, accel=6.0, center_fraction=0.1, seed=123):
    rng = np.random.default_rng(seed)
    ky = np.linspace(-1, 1, ny)
    kx = np.linspace(-1, 1, nx)
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)
    # profile mật độ: cao ở tâm, giảm dần ra rìa
    p = 1/(1 + (R/0.3)**4)
    p /= p.max()

    mask = np.zeros((ny, nx), dtype=bool)
    cy, cx = ny//2, nx//2
    wy, wx = int(ny*center_fraction/2), int(nx*center_fraction/2)
    mask[cy-wy:cy+wy+1, cx-wx:cx+wx+1] = True

    target = 1.0/accel
    curr = mask.mean()
    if curr < target:
        remaining = ~mask
        scale = (target - curr) / (p[remaining].mean() + 1e-12)
        prob = np.clip(p * scale, 0, 1)
        rand = rng.random((ny, nx))
        add = (rand < prob) & remaining
        mask |= add
    return mask, p

def radial_profile(binary_map, nbins=50):
    """Tính trung bình theo vành tròn (annuli) để ra profile theo bán kính."""
    ny, nx = binary_map.shape
    y = np.arange(ny) - ny/2 + 0.5
    x = np.arange(nx) - nx/2 + 0.5
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    r = R.ravel()
    v = binary_map.ravel().astype(float)

    rmax = r.max()
    bins = np.linspace(0, rmax, nbins+1)
    which = np.digitize(r, bins) - 1  # 0..nbins-1
    prof = np.zeros(nbins)
    cnts = np.zeros(nbins)
    for i in range(nbins):
        sel = (which == i)
        cnt = np.sum(sel)
        cnts[i] = cnt
        prof[i] = np.mean(v[sel]) if cnt > 0 else np.nan
    rc = 0.5*(bins[:-1] + bins[1:])
    return rc, prof

def plot_vds_mask(n=256, accel=6.0, center_fraction=0.10, seed=123, nbins=60):
    mask, p = make_var_dens_mask(n, n, accel=accel, center_fraction=center_fraction, seed=seed)
    samp_rate = mask.mean()

    # radial profiles cho p (kỳ vọng) và mask (thực tế)
    _, p_prof = radial_profile(p, nbins=nbins)
    r, m_prof = radial_profile(mask, nbins=nbins)

    fig = plt.figure(figsize=(13,4))

    ax1 = plt.subplot(1,3,1)
    im1 = ax1.imshow(p, cmap="viridis", origin="upper")
    ax1.set_title("Mật độ kỳ vọng p(R)\n(cao ở tâm, giảm dần ra rìa)")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1,3,2)
    ax2.imshow(mask, cmap="gray_r", origin="upper")
    ax2.set_title(f"Mask lấy mẫu (trắng = lấy)\naccel≈{accel:.1f} | rate={samp_rate*100:.1f}%")
    ax2.axis("off")

    ax3 = plt.subplot(1,3,3)
    ax3.plot(r/np.max(r), p_prof, label="Kỳ vọng p(R)")
    ax3.plot(r/np.max(r), m_prof, label="Thực tế mask", alpha=0.8)
    ax3.set_xlabel("Bán kính chuẩn hoá (0=tâm, 1=rìa)")
    ax3.set_ylabel("Tỉ lệ lấy mẫu theo bán kính")
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, ls="--", alpha=0.4)
    ax3.legend()
    ax3.set_title("Hồ sơ xuyên tâm (radial profile)")

    plt.tight_layout()
    plt.show()

# Ví dụ chạy:
if __name__ == "__main__":
    plot_vds_mask(n=256, accel=10.0, center_fraction=0.10, seed=42)
