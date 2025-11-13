from enum import Enum
import numpy as np


class PatternType(Enum):
    BERNOULLI = "bernoulli"
    GAUSSIAN = "gaussian"


class PatternGenerator:
    def __init__(self, img_shape, pattern_type=PatternType.BERNOULLI, seed=None):
        self.pattern_type = pattern_type
        self.seed = seed
        self.N = img_shape[0] * img_shape[1]

    def generate(self, M):
        if self.seed is not None:
            np.random.seed(self.seed)
        if M is None:
            M = self.N / 10

        if self.pattern_type == PatternType.BERNOULLI:
            Φ = np.random.randint(0, 2, size=(M, self.N))
            Φ = Φ * 2 - 1
        elif self.pattern_type == PatternType.GAUSSIAN:
            Φ = np.random.randn(M, self.N)
        else:
            raise ValueError("Unsupported pattern type")
        return Φ
