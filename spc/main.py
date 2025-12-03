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
import time
import pandas as pd
from memory_profiler import memory_usage


def print_resource_usage(tag=""):
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss / (1024**2)  # MB
    cpu_times = p.cpu_times()  # user + system (giây)
    print(
        f"[RES] {tag} | "
        f"Memory RSS = {mem:.2f} MB, "
        f"CPU user={cpu_times.user:.2f}s, system={cpu_times.system:.2f}s"
    )


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


M_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
k_list = [
    0.01,
    0.05,
    0.1,
    0.15,
]
noise_range = [10, 20, 30, 40]
noise_list = ["CLEAN", "GAUSSIAN", "POISSON", "MIXED"]

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

        noise = NoiseGenerator(seed=seed)
        chart_utils = ChartUtils(algo_info={}, figsize=CHART_FIG_SIZE)

        img = read_image(image_dir, image_shape)  # img = data.hubble_deep_field()
        # img = color.rgb2gray(img)
        # img = sk_transform.resize(img, image_shape)
        x = img.ravel()

        N = image_shape[0] * image_shape[1]

        ## running
        coeffs = transform.forward(img)
        x = coeffs.ravel()

        for M_ratio in M_list:
            M = int(M_ratio * N)
            for k_ratio in k_list:
                k = int(k_ratio * N)
                print(f"Measurement M: {M} / {N}, Sparsity level k: {k} / {N}")

                base_path = os.path.join(save_path, f"M_{M_ratio:.2f}_k_{k_ratio:.2f}")
                for noise_type in noise_list:
                    noise.set_noise_type(noise_type)
                    if noise_type == "CLEAN":
                        for algo in algo_list:
                            ssim_mean = 0.0
                            psnr_mean = 0.0
                            mse_mean = 0.0
                            cpu_times = 0.0
                            static_memories = 0.0
                            peak_memories = 0.0

                            seed = config.seed

                            for run_idx in range(config.loop_count):
                                pattern.seed = seed + run_idx
                                A = pattern.generate(M)
                                y = A @ x

                                # không thêm noise
                                algorithms.A = A
                                algorithms.k = k

                                p = psutil.Process(os.getpid())
                                t_wall_start = time.perf_counter()
                                t_cpu_start = time.process_time()

                                x_rec, info = algorithms.reconstruct(
                                    y=y,
                                    algorithm=algo,
                                    tol=config.tol,
                                    max_iter=config.max_iter,
                                    return_info=True,
                                    ignore_iteration_log=False,
                                    measure_memory=True,
                                    x_true=x,
                                    psnr_threshold=20,
                                )

                                t_wall_end = time.perf_counter()
                                t_cpu_end = time.process_time()

                                print(
                                    f"[{algo.value} CLEAN run {run_idx+1}] wall={t_wall_end - t_wall_start:.3f}s, "
                                    f"CPU={t_cpu_end - t_cpu_start:.3f}s"
                                )

                                print(
                                    "Static memory  (MB):",
                                    info.meta["memory_static_MB"],
                                )
                                print(
                                    "Peak memory    (MB):", info.meta["memory_peak_MB"]
                                )
                                print(
                                    "Extra dynamic  (MB):",
                                    info.meta["memory_peak_MB"]
                                    - info.meta["memory_static_MB"],
                                )

                                static_memories += info.meta["memory_static_MB"]
                                peak_memories += info.meta["memory_peak_MB"]
                                cpu_times += t_cpu_end - t_cpu_start

                                coeffs_rec = x_rec.reshape(image_shape)
                                img_rec = transform.inverse(coeffs_rec)

                                ssim_val = ssim(img, img_rec)
                                psnr_val = psnr(img, img_rec, data_range=1.0)
                                mse_val = mse(img, img_rec)

                                ssim_mean += ssim_val
                                psnr_mean += psnr_val
                                mse_mean += mse_val

                                print(
                                    f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, MSE: {mse_val:.6f}"
                                )
                                print("stop reason:", info.stop_reason)
                                if info.iteration_log:
                                    residual_norm = info.history.get(
                                        "residual_norm", []
                                    )
                                    al_path = os.path.join(
                                        base_path,
                                        f"{algo.value}_CLEAN",
                                    )
                                    os.makedirs(al_path, exist_ok=True)
                                    xlsx_path = os.path.join(
                                        al_path,
                                        f"{algo.value}_residuals_run_{run_idx+1}.xlsx",
                                    )
                                    df = pd.DataFrame({"residual_norm": residual_norm})
                                    df.to_excel(xlsx_path, index=False)

                            # average
                            ssim_mean /= config.loop_count
                            psnr_mean /= config.loop_count
                            mse_mean /= config.loop_count
                            cpu_times /= config.loop_count
                            static_memories /= config.loop_count
                            peak_memories /= config.loop_count

                            print(
                                f"[{algo.value} CLEAN AVERAGE] SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}"
                            )
                            print(f"[{algo.value} CLEAN AVERAGE] CPU={cpu_times:.3f}s")

                            summary_path = os.path.join(
                                base_path, f"{algo.value}_CLEAN_summary.txt"
                            )
                            with open(summary_path, "w") as f:
                                f.write(
                                    f"AVERAGE SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}\n"
                                )
                                f.write(f"AVERAGE CPU: {cpu_times:.3f}s\n")
                                f.write(
                                    f"AVERAGE Static Memory (MB): {static_memories:.2f}\n"
                                )
                                f.write(
                                    f"AVERAGE Peak Memory (MB): {peak_memories:.2f}\n"
                                )
                                f.write("stop reason: " + info.stop_reason + "\n")

                    elif noise_type in ("GAUSSIAN", "MIXED"):
                        for snr_db in noise_range:
                            for algo in algo_list:
                                ssim_mean = 0.0
                                psnr_mean = 0.0
                                mse_mean = 0.0
                                cpu_times = 0.0
                                static_memories = 0.0
                                peak_memories = 0.0

                                seed = config.seed

                                for run_idx in range(config.loop_count):
                                    pattern.seed = seed + run_idx
                                    A = pattern.generate(M)
                                    y = A @ x
                                    noise.seed = seed + run_idx
                                    noise.snr_db = snr_db
                                    y_noisy = noise.add_noise(y)

                                    algorithms.A = A
                                    algorithms.k = k

                                    p = psutil.Process(os.getpid())
                                    t_wall_start = time.perf_counter()
                                    t_cpu_start = time.process_time()

                                    x_rec, info = algorithms.reconstruct(
                                        y=y_noisy,
                                        algorithm=algo,
                                        tol=config.tol,
                                        max_iter=config.max_iter,
                                        return_info=True,
                                        ignore_iteration_log=False,
                                        measure_memory=True,
                                        x_true=x,
                                        psnr_threshold=20,
                                    )

                                    t_wall_end = time.perf_counter()
                                    t_cpu_end = time.process_time()

                                    print(
                                        f"[{algo.value} NOISE={noise_type} {snr_db}dB run {run_idx+1}] "
                                        f"wall={t_wall_end - t_wall_start:.3f}s, "
                                        f"CPU={t_cpu_end - t_cpu_start:.3f}s"
                                    )

                                    print(
                                        "Static memory  (MB):",
                                        info.meta["memory_static_MB"],
                                    )
                                    print(
                                        "Peak memory    (MB):",
                                        info.meta["memory_peak_MB"],
                                    )
                                    print(
                                        "Extra dynamic  (MB):",
                                        info.meta["memory_peak_MB"]
                                        - info.meta["memory_static_MB"],
                                    )

                                    static_memories += info.meta["memory_static_MB"]
                                    peak_memories += info.meta["memory_peak_MB"]
                                    cpu_times += t_cpu_end - t_cpu_start

                                    coeffs_rec = x_rec.reshape(image_shape)
                                    img_rec = transform.inverse(coeffs_rec)

                                    ssim_val = ssim(img, img_rec)
                                    psnr_val = psnr(img, img_rec, data_range=1.0)
                                    mse_val = mse(img, img_rec)

                                    ssim_mean += ssim_val
                                    psnr_mean += psnr_val
                                    mse_mean += mse_val

                                    print(
                                        f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, MSE: {mse_val:.6f}"
                                    )
                                    print("stop reason:", info.stop_reason)

                                    if info.iteration_log:
                                        residual_norm = info.history.get(
                                            "residual_norm", []
                                        )
                                        al_path = os.path.join(
                                            base_path,
                                            f"{algo.value}_{noise_type}_{snr_db}dB",
                                        )
                                        os.makedirs(al_path, exist_ok=True)
                                        xlsx_path = os.path.join(
                                            al_path,
                                            f"{algo.value}_residuals_run_{run_idx+1}.xlsx",
                                        )
                                        df = pd.DataFrame(
                                            {"residual_norm": residual_norm}
                                        )
                                        df.to_excel(xlsx_path, index=False)

                                ssim_mean /= config.loop_count
                                psnr_mean /= config.loop_count
                                mse_mean /= config.loop_count
                                cpu_times /= config.loop_count
                                static_memories /= config.loop_count
                                peak_memories /= config.loop_count

                                print(
                                    f"[{algo.value} NOISE={noise_type} {snr_db}dB AVERAGE] "
                                    f"SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}"
                                )
                                print(
                                    f"[{algo.value} NOISE={noise_type} {snr_db}dB AVERAGE] CPU={cpu_times:.3f}s"
                                )

                                summary_path = os.path.join(
                                    base_path,
                                    f"{algo.value}_{noise_type}_{snr_db}dB_summary.txt",
                                )
                                with open(summary_path, "w") as f:
                                    f.write(
                                        f"AVERAGE SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}\n"
                                    )
                                    f.write(f"AVERAGE CPU: {cpu_times:.3f}s\n")
                                    f.write(
                                        f"AVERAGE Static Memory (MB): {static_memories:.2f}\n"
                                    )
                                    f.write(
                                        f"AVERAGE Peak Memory (MB): {peak_memories:.2f}\n"
                                    )
                                    f.write("stop reason: " + info.stop_reason + "\n")
                    elif noise_type == "POISSON":
                        snr_db = None
                        for algo in algo_list:
                            ssim_mean = 0.0
                            psnr_mean = 0.0
                            mse_mean = 0.0
                            cpu_times = 0.0
                            static_memories = 0.0
                            peak_memories = 0.0

                            seed = config.seed

                            for run_idx in range(config.loop_count):
                                pattern.seed = seed + run_idx
                                A = pattern.generate(M)
                                y = A @ x

                                noise.seed = seed + run_idx
                                y_noisy = noise.add_noise(y)

                                algorithms.A = A
                                algorithms.k = k

                                p = psutil.Process(os.getpid())
                                t_wall_start = time.perf_counter()
                                t_cpu_start = time.process_time()

                                x_rec, info = algorithms.reconstruct(
                                    y=y_noisy,
                                    algorithm=algo,
                                    tol=config.tol,
                                    max_iter=config.max_iter,
                                    return_info=True,
                                    ignore_iteration_log=False,
                                    measure_memory=True,
                                    x_true=x,
                                    psnr_threshold=20,
                                )

                                t_wall_end = time.perf_counter()
                                t_cpu_end = time.process_time()

                                print(
                                    f"[{algo.value} NOISE=POISSON run {run_idx+1}] "
                                    f"wall={t_wall_end - t_wall_start:.3f}s, "
                                    f"CPU={t_cpu_end - t_cpu_start:.3f}s"
                                )

                                print(
                                    "Static memory  (MB):",
                                    info.meta["memory_static_MB"],
                                )
                                print(
                                    "Peak memory    (MB):", info.meta["memory_peak_MB"]
                                )
                                print(
                                    "Extra dynamic  (MB):",
                                    info.meta["memory_peak_MB"]
                                    - info.meta["memory_static_MB"],
                                )

                                static_memories += info.meta["memory_static_MB"]
                                peak_memories += info.meta["memory_peak_MB"]
                                cpu_times += t_cpu_end - t_cpu_start

                                coeffs_rec = x_rec.reshape(image_shape)
                                img_rec = transform.inverse(coeffs_rec)

                                ssim_val = ssim(img, img_rec)
                                psnr_val = psnr(img, img_rec, data_range=1.0)
                                mse_val = mse(img, img_rec)

                                ssim_mean += ssim_val
                                psnr_mean += psnr_val
                                mse_mean += mse_val

                                print(
                                    f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, MSE: {mse_val:.6f}"
                                )
                                print("stop reason:", info.stop_reason)

                                if info.iteration_log:
                                    residual_norm = info.history.get(
                                        "residual_norm", []
                                    )
                                    al_path = os.path.join(
                                        base_path,
                                        f"{algo.value}_POISSON",
                                    )
                                    os.makedirs(al_path, exist_ok=True)
                                    xlsx_path = os.path.join(
                                        al_path,
                                        f"{algo.value}_residuals_run_{run_idx+1}.xlsx",
                                    )
                                    df = pd.DataFrame({"residual_norm": residual_norm})
                                    df.to_excel(xlsx_path, index=False)

                            # average cho Poisson
                            ssim_mean /= config.loop_count
                            psnr_mean /= config.loop_count
                            mse_mean /= config.loop_count
                            cpu_times /= config.loop_count
                            static_memories /= config.loop_count
                            peak_memories /= config.loop_count

                            print(
                                f"[{algo.value} NOISE=POISSON AVERAGE] "
                                f"SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}"
                            )
                            print(
                                f"[{algo.value} NOISE=POISSON AVERAGE] CPU={cpu_times:.3f}s"
                            )

                            summary_path = os.path.join(
                                base_path,
                                f"{algo.value}_POISSON_summary.txt",
                            )
                            with open(summary_path, "w") as f:
                                f.write(
                                    f"AVERAGE SSIM: {ssim_mean:.4f}, PSNR: {psnr_mean:.2f} dB, MSE: {mse_mean:.6f}\n"
                                )
                                f.write(f"AVERAGE CPU: {cpu_times:.3f}s\n")
                                f.write(
                                    f"AVERAGE Static Memory (MB): {static_memories:.2f}\n"
                                )
                                f.write(
                                    f"AVERAGE Peak Memory (MB): {peak_memories:.2f}\n"
                                )
                                f.write("stop reason: " + info.stop_reason + "\n")
