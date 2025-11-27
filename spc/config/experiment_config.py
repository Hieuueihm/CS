import json
import ast


# config the experiment
# - noise
# - image size
# - sparsity level
# - transform type
# - loop count
# -
class ExperimentConfig:

    def __init__(self):
        self.image_shape = (64, 64)
        self.sparsity_level = 0.25
        self.transform_type = "FFT"
        self.noise_db = 30
        self.noise_type = "GAUSSIAN"
        self.image_path = "cameraman.jpg"
        self.seed = 42
        self.name = "default_experiment"
        self.pattern_type = "BERNOULLI"
        self.loop_count = 10
        self.measurement_rate = 0.5
        self.tol = 1e-6
        self.max_iter = 1000

    def from_json(self, data):

        if isinstance(data.get("image_shape"), str):
            self.image_shape = tuple(ast.literal_eval(data.get("image_shape")))
        else:
            self.image_shape = tuple(data.get("image_shape", self.image_shape))
        self.sparsity_level = data.get("sparsity_level", self.sparsity_level)
        self.transform_type = data.get("transform_type", self.transform_type)
        self.noise_db = data.get("noise_db", self.noise_db)
        self.pattern_type = data.get(
            "pattern_type", getattr(self, "pattern_type", None)
        )
        self.noise_type = data.get("noise_type", self.noise_type)
        self.image_path = data.get("image_path", self.image_path)
        self.seed = data.get("seed", self.seed)
        self.name = data.get("name", self.name)
        self.loop_count = data.get("loop_count", self.loop_count)
        self.measurement_rate = data.get(
            "measurement_rate", getattr(self, "measurement_rate", None)
        )
        self.tol = data.get("tol", getattr(self, "tol", None))
        self.max_iter = data.get("max_iter", getattr(self, "max_iter", None))

    #       {
    #     "name": "t01",
    #     "image_shape": [
    #         64,
    #         64
    #     ],
    #     "seed": 42,
    #     "pattern_type": "BERNOULLI",
    #     "transform_type": "NONE",
    #     "noise_db": 10,
    #     "noise_type": "GAUSSIAN",
    #     "image_path": "cameraman.jpg",
    #     "loop_count": 20,
    #     "sparsity_level": 0.25
    # }

    # ==============================
    #      PRINT / DEBUG
    # ==============================
    def __repr__(self):
        return (
            "ExperimentConfig("
            f"image_shape={self.image_shape}, "
            f"sparsity_level={self.sparsity_level}, "
            f"transform_type='{self.transform_type}', "
            f"noise_db={self.noise_db}, "
            f"noise_type='{self.noise_type}', "
            f"image_path='{self.image_path}', "
            f"seed={self.seed}, "
            f"loop_count={self.loop_count})",
            f"measurement_rate={self.measurement_rate})",
            f"tol={self.tol})",
            f"max_iter={self.max_iter})",
        )
