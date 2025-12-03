import numpy as np


class NoiseGenerator:
    """
    Noise generator for SPC / compressive sensing.

    Supported noise:
    - Gaussian (signal-independent)
    - Poisson (shot noise)
    - Mixed (Poisson + Gaussian)
    """

    def __init__(self, snr_db=30, noise_type="GAUSSIAN", seed=None):
        self.snr_db = snr_db
        self.noise_type = noise_type.upper()
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def add_noise(self, y_clean):
        if self.noise_type == "GAUSSIAN":
            return self._add_gaussian_noise(y_clean)
        elif self.noise_type == "POISSON":
            return self._add_poisson_noise(y_clean)
        elif self.noise_type == "MIXED":
            return self._add_mixed_noise(y_clean)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def set_noise_type(self, noise_type):
        self.noise_type = noise_type.upper()

    def _add_gaussian_noise(self, y_clean):
        """
        signal-independent noise:
        y_noisy = y + n
        n ~ N(0, sigma^2)
        sigma theo SNR
        """
        signal_power = np.mean(y_clean**2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        sigma = np.sqrt(noise_power)

        noise = sigma * np.random.randn(*y_clean.shape)
        return y_clean + noise

    def _add_poisson_noise(self, y_clean):
        """
        Poisson noise models photon counting:
        y_clean >= 0, then y_noisy ~ Poisson(lambda = scale * y_clean)
        Must rescale intensity so lambda is not tiny.
        """
        # Ensure non-negative intensities
        y = np.maximum(y_clean, 0)

        # Scale intensity to photon count level
        # 1000 photons is common for simulation
        scale = 1000
        lam = scale * y

        # Poisson sampling
        y_poiss = np.random.poisson(lam)

        # Rescale back to original range
        y_noisy = y_poiss / scale
        return y_noisy

    # -----------------------------------------------------------------------
    # 3. Mixed noise (Poisson + Gaussian)
    # -----------------------------------------------------------------------
    def _add_mixed_noise(self, y_clean):
        y_poiss = self._add_poisson_noise(y_clean)
        y_final = self._add_gaussian_noise(y_poiss)
        return y_final
