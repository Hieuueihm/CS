import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fftpack import dct, idct
from sklearn.linear_model import OrthogonalMatchingPursuit
import time
from cosamp import cs_cosamp

def demo_cs_omp(image_path="lena.bmp"):
    # Đọc ảnh grayscale
    img = io.imread(image_path, as_gray=True)
    img = img * 255.0
    height, width = img.shape

    # Measurement matrix Phi
    m = height // 3
    Phi = np.random.randn(m, height)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)

    # DCT basis (1D)
    mat_dct_1d = np.zeros((height, height))
    for k in range(height):
        dct_vec = np.cos(np.arange(height) * k * np.pi / height)
        if k > 0:
            dct_vec -= np.mean(dct_vec)
        mat_dct_1d[:, k] = dct_vec / np.linalg.norm(dct_vec)

    # Projection
    img_cs_1d = Phi @ img

    # Recover using OMP
    sparse_rec_1d = np.zeros((height, width))
    Theta_1d = Phi @ mat_dct_1d
    start = time.time()
	# Recover từng cột bằng OMP
    for i in range(width):
        # y = img_cs_1d[:, i]        # measurements
        # omp = OrthogonalMatchingPursuit(n_nonzero_coefs=height//4)  # chọn độ thưa\
        # omp.fit(Theta_1d, y)
        # coef = omp.coef_
        # sparse_rec_1d[:, i] = coef
        column_rec, it = cs_cosamp(img_cs_1d[:, i], Theta_1d, height // 10)
        sparse_rec_1d[:, i] = column_rec.T


	# Reconstruct ảnh
    img_rec_1d = mat_dct_1d @ sparse_rec_1d
    end = time.time()
	# PSNR
    rec_psnr = psnr(img, img_rec_1d, data_range=img.max() - img.min())

	# Hiển thị kết quả
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image"), plt.axis("off")
    plt.subplot(2, 2, 2), plt.imshow(Phi, cmap="gray"), plt.title("Measurement Matrix"), plt.axis("off")
    plt.subplot(2, 2, 3), plt.imshow(mat_dct_1d, cmap="gray"), plt.title("1D DCT Basis"), plt.axis("off")
    plt.subplot(2, 2, 4), plt.imshow(img_rec_1d, cmap="gray"), plt.title(f"Reconstructed ({rec_psnr:.2f} dB)"), plt.axis("off")
    plt.show()
    print(f"time {end - start}")


#------------------------------
# Run Demo
#------------------------------
demo_cs_omp("../lena.bmp") 