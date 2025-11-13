class LinearOperator:
    def __init__(self, forward, adjoint, shape):
        self.forward = forward
        self.adjoint = adjoint
        self.shape = shape  # (M, N)
