import numpy as np
import os
import sys
import json
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from skimage import data, color, transform as sk_transform
import psutil
import math
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd


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


EXPERIMENT_FOLDER = "experiments"
RESULTS_FOLDER = "results"
DATA_FOLDER = "data"
CHART_FIG_SIZE = (4, 4)


algo_list = [
    ReconstructionAlgorithms.OMP,
    ReconstructionAlgorithms.IHT,
    ReconstructionAlgorithms.COSAMP,
    ReconstructionAlgorithms.SP,
    ReconstructionAlgorithms.ISTA,
    ReconstructionAlgorithms.FISTA,
]


def read_image(filepath, size):
    img = Image.open(filepath).convert("L")  # convert sang grayscale
    img_resized = img.resize(size)
    x = np.array(img_resized, dtype=np.float32)
    x = x / 255.0
    return x


if __name__ == "__main__":
    pin_process()

    config = ExperimentConfig()
    with open(os.path.join(EXPERIMENT_FOLDER, "test_case.json"), "r") as f:
        tc = json.load(f)
    for _ in range(len(tc)):
        config.from_json(tc[_])
        tc_name = config.name
        image_shape = config.image_shape
        seed = config.seed
        pattern_type = config.pattern_type
        transform_type = config.transform_type
        noise_db = config.noise_db
        noise_type = config.noise_type
        image_path = config.image_path
        save_path = os.path.join(RESULTS_FOLDER, f"{tc_name}")
        os.makedirs(save_path, exist_ok=True)

        print(
            f"Running test case {_ + 1}/{len(tc)}: image_shape={image_shape} image_path = {image_path}"
        )
        image_dir = os.path.join(DATA_FOLDER, image_path)

        algorithms = BaseReconstructor(N=image_shape[0] * image_shape[1])
        pattern = PatternGenerator(img_shape=image_shape, seed=seed)

        if pattern_type == "BERNOULLI":
            pattern.set_pattern_type(PatternType.BERNOULLI)
        elif pattern_type == "GAUSSIAN":
            pattern.set_pattern_type(PatternType.GAUSSIAN)
        else:
            raise ValueError("Unsupported pattern type")
        print(
            f"Using pattern type: {pattern_type} and transform type: {transform_type}"
        )
        pattern.set_transform_type(transform_type)

        transform = Transform(img_shape=image_shape)
        if transform_type == "DCT":
            transform.set_transfrom_kind(TransformKind.DCT)
        elif transform_type == "FFT":
            transform.set_transfrom_kind(TransformKind.FFT)
        elif transform_type == "NONE":
            transform.set_transfrom_kind(TransformKind.NONE)
        else:
            raise ValueError("Unsupported transform type")

        noise = NoiseGenerator(noise_type=noise_type, seed=seed)
        chart_utils = ChartUtils(algo_info={}, figsize=CHART_FIG_SIZE)

        img = read_image(image_dir, image_shape)  # img = data.hubble_deep_field()
        # img = color.rgb2gray(img)
        # img = sk_transform.resize(img, image_shape)
        x = img.ravel()

        N = image_shape[0] * image_shape[1]

        M = math.floor(config.measurement_rate * N)
        print(f"Number of measurements M: {M}")

        ## running
        coeffs = transform.forward(img)
        x = coeffs.ravel()

        for algo in algo_list:
            for i in range(config.loop_count):
                pattern.seed = seed + i
                A = pattern.generate(M)
                y = A @ x
                algorithms.A = A
                algorithms.k = int(config.sparsity_level * M)

                x_rec, info = algorithms.reconstruct(
                    y=y, algorithm=algo, tol=config.tol, max_iter=config.max_iter
                )

                # img_rec = x_rec.reshape(image_shape)
                coeffs_rec = x_rec.reshape(image_shape)
                img_rec = transform.inverse(coeffs_rec)

                ssim_val = ssim(img, img_rec)
                psnr_val = psnr(img, img_rec, data_range=1.0)
                mse_val = mse(img, img_rec)
                print(
                    f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, MSE: {mse_val:.6f}"
                )
                # chart_utils.plot_image(
                #     img_rec,
                #     title=f"{algo} Reconstruction",
                #     save_path=None,
                # )

                residual_norm = info.history.get("residual_norm", [])

                al_path = os.path.join(save_path, f"{algo.value}")
                os.makedirs(al_path, exist_ok=True)

                xlsx_path = os.path.join(
                    al_path, f"{algo.value}_residuals_run_{i+1}.xlsx"
                )

                df = pd.DataFrame({"residual_norm": residual_norm})
                df.to_excel(xlsx_path, index=False)
                # chart_utils.plot_convergence(
                #     residuals=residual_norm, title=f"{algo} Convergence", save_path=None
                # )

                # break
            break
