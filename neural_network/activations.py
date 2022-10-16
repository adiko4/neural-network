import numpy as np
from typing import Protocol, Any


class ActivationFunction(Protocol):
    """Protocol class for a neural network activation function"""

    @staticmethod
    def calculate(z) -> Any:
        ...

    @staticmethod
    def derivative(z) -> Any:
        ...


class SigmoidActivation:
    """Sigmoid activation function"""

    @staticmethod
    def calculate(z) -> Any:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z) -> Any:
        a = SigmoidActivation.calculate(z)
        return a * (1 - a)


class ReLUActivation:
    """ReLU activation function"""

    @staticmethod
    def calculate(z) -> Any:
        return np.maximum(0, z)

    @staticmethod
    def derivative(z) -> Any:
        derivatives = z.copy()
        derivatives[derivatives >= 0] = 1
        derivatives[derivatives < 0] = 0
        return derivatives
