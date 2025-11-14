class AlgorithmInformation:
    OMP_STOP_INFORMATION = "omp_stop_information"
    IHT_STOP_INFORMATION = "iht_stop_information"
    ISTA_STOP_INFORMATION = "fista_stop_information"
    FISTA_STOP_INFORMATION = "fista_stop_information"
    COSAMP_STOP_INFORMATION = "cosamp_stop_information"
    SP_STOP_INFORMATION = "sp_stop_information"

    def __init__(self, algorithm_name, iteration_log=True):
        """
        algorithm_name: ex 'omp', 'iht', 'fista'
        """
        self.algorithm_name = algorithm_name

        self.stop_reason = None

        self.num_iterations = 0
        self.iteration_log = iteration_log
        self.history = {
            "residual_norm": [],  # ||r_k||
            "sparsity": [],  # số phần tử khác 0
            "step_size": [],  #  gradient / learning rate
            "support": [],  # support (OMP, IHT)
        }

        self.meta = {}

    def add_iteration(
        self,
        residual_norm,
        sparsity,
        step_size,
        support,
    ):
        if residual_norm is not None:
            self.history["residual_norm"].append(float(residual_norm))
        else:
            pass

        if sparsity is not None:
            self.history["sparsity"].append(int(sparsity))

        if step_size is not None:
            self.history["step_size"].append(float(step_size))

        if support is not None:
            self.history["support"].append(support)

        self.num_iterations += 1

    def set_stop_reason(self, reason):
        self.stop_reason = reason

    def set_meta(self, key, value):
        self.meta[key] = value

    def set_time(self, time):
        self.time = time

    def get(self):
        return {
            "algorithm_name": self.algorithm_name,
            "stop_reason": self.stop_reason,
            "num_iterations": self.num_iterations,
            "history": self.history,
            "meta": self.meta,
            "time": self.time,
        }
