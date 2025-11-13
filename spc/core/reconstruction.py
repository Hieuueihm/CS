from enum import Enum


class ReconstructionAlgorithms(Enum):
    OMP = "omp"
    IHT = "iht"
    FISTA = "fista"


class BaseReconstructor:
    def __init__(self, operator, algorithm=ReconstructionAlgorithms.OMP, **kwargs):
        self.operator = operator
        self.algorithm = algorithm
        self.params = kwargs

    def reconstruct(self, y):
        if self.algorithm == ReconstructionAlgorithms.OMP:
            return self._reconstruct_omp(y)
        elif self.algorithm == ReconstructionAlgorithms.IHT:
            return self._reconstruct_iht(y)
        elif self.algorithm == ReconstructionAlgorithms.FISTA:
            return self._reconstruct_fista(y)
        else:
            raise ValueError(f"Unknown reconstruction algorithm: {self.algorithm}")

    def _reconstruct_omp(self, measurements):
        raise NotImplementedError("OMP reconstruction not implemented yet")

    def _reconstruct_iht(self, measurements):
        raise NotImplementedError("IHT reconstruction not implemented yet")

    def _reconstruct_fista(self, measurements):
        raise NotImplementedError("FISTA reconstruction not implemented yet")
