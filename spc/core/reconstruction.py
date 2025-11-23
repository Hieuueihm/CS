from enum import Enum
import sys
import numpy as np


sys.path.append("algorithms/")
sys.path.append("../config/")
from algorithms.cosamp import cs_cosamp
from algorithms.fista import cs_fista
from algorithms.ista import cs_ista
from algorithms.sp import cs_sp
from algorithms.omp import cs_omp
from algorithms.iht import cs_iht
from config.experiment_config import ExperimentConfig


class ReconstructionAlgorithms(Enum):
    OMP = "omp"
    IHT = "iht"
    FISTA = "fista"
    ISTA = "ista"
    COSAMP = "cosamp"
    SP = "subspace_pursuit"


class BaseReconstructor:
    def __init__(
        self,
        N=None,
        A=None,
        k=None,
        params=None,
    ):
        self.N = N if N is not None else 64 * 64
        self.A = A
        self.k = k if k is not None else self.N // 4
        self.params = params if params is not None else ExperimentConfig()

    def reconstruct(self, y, algorithm=None, **kwargs):
        return_info = kwargs.get("return_info", True)
        ignore_iteration_log = kwargs.get("ignore_iteration_log", False)
        if algorithm is not None:
            self.algorithm = algorithm
        if self.algorithm == ReconstructionAlgorithms.OMP:
            return self._reconstruct_omp(y, return_info, ignore_iteration_log)
        elif self.algorithm == ReconstructionAlgorithms.IHT:
            return self._reconstruct_iht(y, return_info, ignore_iteration_log)
        elif self.algorithm == ReconstructionAlgorithms.FISTA:
            return self._reconstruct_fista(y, return_info, ignore_iteration_log)
        elif self.algorithm == ReconstructionAlgorithms.ISTA:
            return self._reconstruct_ista(y, return_info, ignore_iteration_log)
        elif self.algorithm == ReconstructionAlgorithms.COSAMP:
            return self._reconstruct_cosamp(y, return_info, ignore_iteration_log)
        elif self.algorithm == ReconstructionAlgorithms.SP:
            return self._reconstruct_sp(y, return_info, ignore_iteration_log)
        else:
            raise ValueError(f"Unknown reconstruction algorithm: {self.algorithm}")

    def _reconstruct_omp(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_omp(
            measurement_vector,
            self.A,
            self.k,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )

    def _reconstruct_ista(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_ista(
            measurement_vector,
            self.A,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )

    def _reconstruct_fista(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_fista(
            measurement_vector,
            self.A,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )

    def _reconstruct_cosamp(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_cosamp(
            measurement_vector,
            self.A,
            self.k,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )

    def _reconstruct_sp(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_sp(
            measurement_vector,
            self.A,
            self.k,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )

    def _reconstruct_iht(
        self, measurement_vector, return_info=True, ignore_iteration_log=False
    ):
        return cs_iht(
            measurement_vector,
            self.A,
            self.k,
            return_info=return_info,
            ignore_iteration_log=ignore_iteration_log,
        )
