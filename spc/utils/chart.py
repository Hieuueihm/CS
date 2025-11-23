import numpy as np
import matplotlib.pyplot as plt
import os


class ChartUtils:
    def __init__(self, algo_info, figsize=(4, 4)):
        self.info = algo_info
        self.figsize = figsize

    def summary(self):
        pass

    def save_json(self):
        pass

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

        # Nếu là vector → reshape về ảnh vuông
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
