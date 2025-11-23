import numpy as np
import os
import sys
import json
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from skimage import data, color, transform as sk_transform
import psutil
import math


def pin_process():
    p = psutil.Process(os.getpid())

    # 1) Ghim vào core 0 (có thể chọn core khác, ví dụ [1], [2], ...)
    try:
        p.cpu_affinity([0])
    except AttributeError:
        # cpu_affinity không có trên 1 số hệ điều hành, nhưng Windows thì có
        pass

    # 2) Tăng priority (Windows)
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except AttributeError:
        pass


sys.path.append("config/")
sys.path.append("core/")
sys.path.append("utils/")
sys.path.append("experiments/")
from config.experiment_config import ExperimentConfig
from core.reconstruction import BaseReconstructor, ReconstructionAlgorithms
from utils.chart import ChartUtils
from core.metrics import mse, psnr, ssim
from core.patterns import PatternGenerator, PatternType
from PIL import Image
from core.transform import Transform, TransformKind
from core.noise import NoiseGenerator


def load_astronomy_image(img_shape=(64, 64), normalize=True):
    img = data.hubble_deep_field()
    img = color.rgb2gray(img)

    img = sk_transform.resize(img, img_shape)

    arr = img.astype(np.float32)

    if normalize:
        arr = arr / arr.max()

    return arr


IMAGE_SHAPE = (64, 64)
SEED = 42
EXPERIMENT_FOLDER = "experiments"
DATA_FOLDER = "data"
FIG_SIZE = (6, 6)


def load_mnist_digit(index=0, img_shape=IMAGE_SHAPE, normalize=True):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist["data"]
    y = mnist["target"]

    img28 = X[index].reshape(28, 28).astype(np.uint8)

    pil_img = Image.fromarray(img28).resize(img_shape, Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32)

    if normalize:
        arr = arr / 255.0
    return arr, int(y[index])


def read_image(filepath, size):
    img = Image.open(filepath).convert("L")  # convert sang grayscale
    img_resized = img.resize(size)
    x = np.array(img_resized, dtype=np.float32)
    x = x / 255.0
    return x


if __name__ == "__main__":
    pin_process()

    config = ExperimentConfig()
    algorithms = BaseReconstructor(N=IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    pattern = PatternGenerator(
        IMAGE_SHAPE, pattern_type=PatternType.BERNOULLI, seed=SEED
    )
    noise = NoiseGenerator(noise_type="gaussian", seed=SEED)
    chart_utils = ChartUtils(algo_info={}, figsize=FIG_SIZE)
    # img = read_image(os.path.join(DATA_FOL
    # DER, "lena.png"), IMAGE_SHAPE)
    # chart_utils.plot_image(img, title="Original Image", save_path=None)
    transform = Transform(img_shape=IMAGE_SHAPE, kind=TransformKind.FFT)

    # coeffs = transform.forward(img)

    pattern.set_pattern_type(PatternType.GAUSSIAN)
    # img, label = load_mnist_digit(index=1, img_shape=IMAGE_SHAPE)
    # chart_utils.plot_image(img, title="Original Image", save_path=None)

    # print("Digit label:", label)
    # print("Shape:", img.shape)
    # chart_utils.plot_frequency(
    #     coeffs,
    #     title="Frequency Domain (FFT Magnitude)",
    #     save_path=None,
    # )

    img = load_astronomy_image(IMAGE_SHAPE)
    # chart_utils.plot_image(img, title="Astronomy Image")

    x = img.reshape(-1)

    with open(os.path.join(EXPERIMENT_FOLDER, "test_case.json"), "r") as f:
        tc = json.load(f)

    for _ in range(len(tc)):
        config.from_json(tc[_])
        A = pattern.generate(
            M=int(config.sparsity_level * IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
        )

        snr_db = config.noise_db
        noise.snr_db = snr_db

        y_clean = A @ x
        y_noisy = noise.add_fixed_noise(y_clean)

        # reconstructor
        algorithms.A = A
        algorithms.k = math.floor(config.sparsity_level * x.shape[0])
        x_rec, info = algorithms.reconstruct(
            y=y_clean, algorithm=ReconstructionAlgorithms.OMP
        )
        print(info.get())
        x_rec_img = x_rec.reshape(IMAGE_SHAPE)
        snr_db = psnr(x, x_rec.astype(np.float32))

        chart_utils.plot_image(
            x_rec_img,
            title=f"Reconstructed Image (SNR={snr_db} dB)",
            save_path=None,
        )
