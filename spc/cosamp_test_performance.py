import os
import numpy as np
import sys
from PIL import Image
import psutil
import time

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


RESULTS_FOLDER = "results"


def read_image(filepath, size):
    img = Image.open(filepath).convert("L")  # convert sang grayscale
    img_resized = img.resize(size)
    x = np.array(img_resized, dtype=np.float32)
    x = x / 255.0
    return x


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


algo_list = [
    ReconstructionAlgorithms.COSAMP,
]
tol_list = [1e-06, 3.16e-06, 1e-05, 3.16e-05, 1e-04, 3.16e-04, 1e-03]
max_iter_list = [50, 100, 200, 300, 500, 800, 1000]
DATA_FOLDER = "data"
image_path = "mnist_cs/mnist_0001.png"
M_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
k_list = [0.01, 0.05, 0.1, 0.15]

if __name__ == "__main__":
    pin_process()
    cosamp_path = os.path.join(RESULTS_FOLDER, "cosamp_test")
    image_dir = os.path.join(DATA_FOLDER, image_path)
    image_shape = (64, 64)
    seed = 42
    algorithms = BaseReconstructor(N=image_shape[0] * image_shape[1])
    pattern = PatternGenerator(img_shape=image_shape, seed=seed)
    pattern.set_pattern_type(PatternType.GAUSSIAN)
    pattern.set_transform_type("DCT")
    transform = Transform(img_shape=image_shape)
    transform.set_transfrom_kind(TransformKind.DCT)
    img = read_image(image_dir, image_shape)
    x = img.ravel()
    os.makedirs(cosamp_path, exist_ok=True)

    N = image_shape[0] * image_shape[1]
    for M_ratio in M_list:
        M = int(M_ratio * N)
        for k_ratio in k_list:
            k = int(k_ratio * N)
            print(f"Measurement M: {M} / {N}, Sparsity level k: {k} / {N}")
            coeffs = transform.forward(img)
            x = coeffs.ravel()
            A = pattern.generate(M)
            A = A / np.linalg.norm(A, axis=0, keepdims=True)
            y = A @ x
            size_path = os.path.join(cosamp_path, f"{M_ratio}_{k_ratio}")
            os.makedirs(size_path, exist_ok=True)
            algorithms.A = A
            algorithms.k = k
            seed = seed + 1
            pattern.seed = seed
            for tol in tol_list:
                for max_iter in max_iter_list:
                    for algo in algo_list:
                        static_memories = 0
                        peak_memories = 0
                        cpu_times = 0
                        p = psutil.Process(os.getpid())
                        t_wall_start = time.perf_counter()
                        t_cpu_start = time.process_time()

                        x_rec, info = algorithms.reconstruct(
                            y=y,
                            algorithm=algo,
                            tol=tol,
                            max_iter=max_iter,
                            return_info=True,
                            ignore_iteration_log=False,
                            measure_memory=True,
                        )

                        t_wall_end = time.perf_counter()
                        t_cpu_end = time.process_time()

                        summary_path = os.path.join(
                            size_path,
                            f"{algo.value}_{tol}_{max_iter}_summary.txt",
                        )
                        coeffs_rec = x_rec.reshape(image_shape)
                        img_rec = transform.inverse(coeffs_rec)

                        ssim_val = ssim(img, img_rec)
                        psnr_val = psnr(img, img_rec, data_range=1.0)
                        mse_val = mse(img, img_rec)
                        print(
                            f"[{algo.value} run wall={t_wall_end - t_wall_start:.3f}s, "
                            f"CPU={t_cpu_end - t_cpu_start:.3f}s "
                            f"psnr={psnr_val}"
                        )
                        static_memories += info.meta["memory_static_MB"]
                        peak_memories += info.meta["memory_peak_MB"]
                        cpu_times += t_cpu_end - t_cpu_start
                        with open(summary_path, "w") as f:
                            f.write(
                                f"AVERAGE SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, MSE: {mse_val:.6f}\n"
                            )
                            f.write(f"AVERAGE CPU: {cpu_times:.3f}s\n")
                            f.write(
                                f"AVERAGE Static Memory (MB): {static_memories:.2f}\n"
                            )
                            f.write(f"AVERAGE Peak Memory (MB): {peak_memories:.2f}\n")
                            f.write("stop reason: " + info.stop_reason + "\n")
