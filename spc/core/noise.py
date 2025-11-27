import numpy as np


class NoiseGenerator:
    def __init__(self, snr_db=30, noise_type="GAUSSIAN", seed=None):

        self.snr_db = snr_db
        self.noise_type = noise_type

        if seed is not None:
            np.random.seed(seed)

    def add_fixed_noise(self, y_clean):

        signal_power = np.mean(y_clean**2)

        noise_power = signal_power / (10 ** (self.snr_db / 10))

        if self.noise_type == "GAUSSIAN":
            noise = np.sqrt(noise_power) * np.random.randn(*y_clean.shape)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

        y_noisy = y_clean + np.abs(noise)

        return y_noisy
