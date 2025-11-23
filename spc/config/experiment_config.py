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

    def __init__(
        self,
        image_size=None,
        sparsity_level=None,
        transform_type=None,
        noise_db=None,
        loop_count=None,
    ):

        self.image_size = image_size if image_size is not None else (64, 64)
        self.sparsity_level = sparsity_level if sparsity_level is not None else 0.1
        self.transform_type = transform_type if transform_type is not None else "FFT"
        self.noise_db = noise_db if noise_db is not None else 0.01
        self.loop_count = loop_count if loop_count is not None else 10

    def from_json(self, data):

        if isinstance(data.get("image_size"), str):
            self.image_size = tuple(ast.literal_eval(data.get("image_size")))
        else:
            self.image_size = tuple(data.get("image_size", self.image_size))

        self.sparsity_level = data.get("sparsity_level", self.sparsity_level)
        self.transform_type = data.get("transform_type", self.transform_type)
        self.noise_db = data.get("noise_db", self.noise_db)
        self.loop_count = data.get("loop_count", self.loop_count)

    def to_json(self, filepath):
        config = {
            "image_size": list(self.image_size),
            "sparsity_level": self.sparsity_level,
            "transform_type": self.transform_type,
            "noise_db": self.noise_db,
            "loop_count": self.loop_count,
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)

    # ==============================
    #      PRINT / DEBUG
    # ==============================
    def __repr__(self):
        return (
            "ExperimentConfig("
            f"image_size={self.image_size}, "
            f"sparsity_level={self.sparsity_level}, "
            f"transform_type='{self.transform_type}', "
            f"noise_db={self.noise_db}, "
            f"loop_count={self.loop_count})"
        )
