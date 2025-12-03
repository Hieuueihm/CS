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
        self.transform_type = "FFT"
        self.image_path = "cameraman.jpg"
        self.seed = 42
        self.name = "default_experiment"
        self.pattern_type = "BERNOULLI"
        self.loop_count = 10
        self.tol = 1e-6
        self.max_iter = 1000

    def from_json(self, data):

        if isinstance(data.get("image_shape"), str):
            self.image_shape = tuple(ast.literal_eval(data.get("image_shape")))
        else:
            self.image_shape = tuple(data.get("image_shape", self.image_shape))
        self.transform_type = data.get("transform_type", self.transform_type)
        self.pattern_type = data.get(
            "pattern_type", getattr(self, "pattern_type", None)
        )
        self.image_path = data.get("image_path", self.image_path)
        self.seed = data.get("seed", self.seed)
        self.name = data.get("name", self.name)
        self.loop_count = data.get("loop_count", self.loop_count)
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
            f"transform_type='{self.transform_type}', "
            f"image_path='{self.image_path}', "
            f"seed={self.seed}, "
            f"loop_count={self.loop_count})",
            f"tol={self.tol})",
            f"max_iter={self.max_iter})",
        )
