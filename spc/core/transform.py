from enum import Enum
from scipy.fft import dctn, idctn, fftn, ifftn
import numpy as np


class TransformKind(Enum):
    DCT = "dct"
    FFT = "fft"
    NONE = "no_transform"


class Transform:
    def __init__(self, img_shape, kind=TransformKind.DCT):
        self.N = img_shape[0] * img_shape[1]
        self.kind = kind
        self.img_shape = img_shape

    def forward(self, x):
        img = x.reshape(self.img_shape)

        if self.kind == TransformKind.DCT:
            return self._forward_dct(img)

        elif self.kind == TransformKind.FFT:
            return self._forward_fft(img)
        else:
            raise ValueError(f"Unsupported transform kind: {self.kind}")

    def inverse(self, coeffs):
        if self.kind == TransformKind.DCT:
            img = self._inverse_dct(coeffs)

        elif self.kind == TransformKind.FFT:
            img = self._inverse_fft(coeffs)

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
