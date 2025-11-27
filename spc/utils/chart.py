import numpy as np
import matplotlib.pyplot as plt
import os


class ChartUtils:
    def __init__(self, algo_info, figsize=(4, 4)):
        self.info = algo_info
        self.figsize = figsize

    def plot_image(self, img, title="Image", cmap="gray", figsize=None, save_path=None):

        arr = np.array(img)
        if arr.ndim == 1:
            side = int(np.sqrt(len(arr)))
            if side * side == len(arr):
                arr = arr.reshape((side, side))
            else:
                raise ValueError("Cannot reshape vector to square image.")

        plt.figure(figsize=figsize if figsize else self.figsize)
        plt.imshow(arr, cmap=cmap)
        plt.title(title)
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    def plot_frequency(
        self,
        coeffs,
        title="Frequency Domain",
        cmap="gray",
        figsize=None,
        save_path=None,
        log_scale=True,
    ):

        arr = np.array(coeffs)

        if arr.ndim == 1:
            side = int(np.sqrt(len(arr)))
            if side * side == len(arr):
                arr = arr.reshape((side, side))
            else:
                raise ValueError("Cannot reshape vector to square 'frequency map'.")

        mag = np.abs(arr)

        mag = np.fft.fftshift(mag)
        if log_scale:
            mag = np.log1p(mag)

        plt.figure(figsize=figsize if figsize else self.figsize)
        plt.imshow(mag, cmap=cmap)
        plt.title(title)
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()

    def plot_convergence(
        self,
        residuals=None,
        title="Convergence Curve",
        figsize=None,
        save_path=None,
        log_scale=True,
    ):

        # Extract residual norms
        # residuals = [it["residual_norm"] for it in self.info.iterations]
        residuals = residuals if residuals is not None else []
        plt.figure(figsize=figsize if figsize else self.figsize)

        if log_scale:
            plt.semilogy(residuals, marker="o", markersize=3)
        else:
            plt.plot(residuals, marker="o", markersize=3)

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Residual Norm")
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()
