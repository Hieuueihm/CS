from enum import Enum
import numpy as np
from scipy.fftpack import dct, idct


class PatternType(Enum):
    BERNOULLI = "BERNOULLI"
    GAUSSIAN = "GAUSSIAN"


class PatternGenerator:
    def __init__(
        self,
        img_shape,
        pattern_type=PatternType.BERNOULLI,
        transform_type="NONE",
        seed=None,
    ):
        self.pattern_type = pattern_type
        self.transform_type = transform_type
        self.seed = seed
        self.H, self.W = img_shape
        self.N = self.H * self.W

    def set_pattern_type(self, pattern_type):
        self.pattern_type = pattern_type

    def set_transform_type(self, transform_type):
        self.transform_type = transform_type

    # ---------------------------------------------------------------------
    # Generate spatial-domain random patterns
    # ---------------------------------------------------------------------
    def _generate_spatial_A(self, M):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.pattern_type == PatternType.BERNOULLI:
            A = np.random.randint(0, 2, size=(M, self.N))
            A = A * 2 - 1  # {-1, +1}

        elif self.pattern_type == PatternType.GAUSSIAN:
            A = np.random.randn(M, self.N)

        else:
            raise ValueError("Unsupported pattern type")

        return A.astype(np.float64)

    # ---------------------------------------------------------------------
    # Build FFT & IFFT operator matrices
    # ---------------------------------------------------------------------
    def _build_fft2_matrix(self):
        H, W = self.H, self.W
        N = H * W
        F = np.zeros((N, N), dtype=np.complex128)

        for j in range(N):
            e = np.zeros((H, W), dtype=np.complex128)
            e.ravel()[j] = 1.0
            E = np.fft.fft2(e, norm="ortho")
            F[:, j] = E.ravel()

        return F

    def _build_ifft2_matrix(self):
        F = self._build_fft2_matrix()
        return F.conj().T

    # ---------------------------------------------------------------------
    # Build DCT and IDCT operator matrices (real + orthonormal)
    # ---------------------------------------------------------------------
    def _build_dct2_matrix(self):
        """
        Build orthonormal DCT-II matrix of shape (N, N)
        Using scipy.fftpack.dct with norm='ortho'
        """
        H, W = self.H, self.W
        N = H * W
        D = np.zeros((N, N), dtype=np.float64)

        for j in range(N):
            e = np.zeros((H, W), dtype=np.float64)
            e.ravel()[j] = 1.0
            E = dct(dct(e.T, norm="ortho").T, norm="ortho")  # 2D DCT-II
            D[:, j] = E.ravel()

        return D

    def _build_idct2_matrix(self):
        """
        Build IDCT-II matrix = inverse of DCT-II = D^T (because orthonormal)
        """
        D = self._build_dct2_matrix()
        return D.T  # orthonormal DCT ⇒ inverse = transpose

    # ---------------------------------------------------------------------
    # Apply transform to A
    # ---------------------------------------------------------------------
    def _apply_transform(self, A):
        if self.transform_type == "NONE":
            return A

        elif self.transform_type == "FFT":
            F_H = self._build_ifft2_matrix()  # (N, N)
            Φ = A @ F_H
            return Φ

        elif self.transform_type == "DCT":
            D_H = self._build_idct2_matrix()  # (N, N)
            Φ = A @ D_H
            return Φ

        else:
            raise ValueError("Unsupported transform type")

    # ---------------------------------------------------------------------
    # Public generate function
    # ---------------------------------------------------------------------
    def generate(self, M):
        if M is None:
            M = self.N // 10  # fallback nếu không truyền M

        A = self._generate_spatial_A(M)
        Φ = self._apply_transform(A)

        return Φ
