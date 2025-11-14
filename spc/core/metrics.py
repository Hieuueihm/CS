import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def mse(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"MSE: shape mismatch {x.shape} vs {y.shape}")

    return mean_squared_error(x, y)


def psnr(x, y, data_range=None):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"PSNR: shape mismatch {x.shape} vs {y.shape}")

    return peak_signal_noise_ratio(x, y, data_range=data_range)


def ssim(x, y, data_range=None):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"SSIM: shape mismatch {x.shape} vs {y.shape}")

    if x.ndim == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    if data_range is None:
        data_range = x.max() - x.min()
        if data_range == 0:
            data_range = 1.0

    return structural_similarity(x, y, data_range=data_range)
