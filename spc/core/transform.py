from enum import Enum
from scipy.fft import dctn, idctn, fftn, ifftn
import numpy as np


class TransformKind(Enum):
    DCT = "DCT"
    FFT = "FFT"
    NONE = "NO_TRANSFORM"


class Transform:
    def __init__(self, img_shape, kind=TransformKind.DCT):
        self.N = img_shape[0] * img_shape[1]
        self.kind = kind
        self.img_shape = img_shape

    def set_transfrom_kind(self, kind):
        self.kind = kind

    def forward(self, x):
        img = x.reshape(self.img_shape)

        if self.kind == TransformKind.DCT:
            return self._forward_dct(img)

        elif self.kind == TransformKind.FFT:
            return self._forward_fft(img)
        elif self.kind == TransformKind.NONE:
            return img
        else:
            raise ValueError(f"Unsupported transform kind: {self.kind}")

    def flatten_forward(self, x):
        coeffs = self.forward(x)
        return coeffs.reshape(-1)

    def inverse(self, coeffs):
        if self.kind == TransformKind.DCT:
            img = self._inverse_dct(coeffs)

        elif self.kind == TransformKind.FFT:
            img = self._inverse_fft(coeffs)
        elif self.kind == TransformKind.NONE:
            img = coeffs
        else:
            raise ValueError(f"Unsupported transform kind: {self.kind}")

        return img.reshape(self.img_shape)

    def _forward_dct(self, img):
        return dctn(img, type=2, norm="ortho")

    def _inverse_dct(self, coeffs):
        return idctn(coeffs, type=2, norm="ortho")

    def _forward_fft(self, img):
        return fftn(img)

    def _inverse_fft(self, coeffs):
        return np.real(ifftn(coeffs))
