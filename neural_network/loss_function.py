import numpy as np
from typing import Protocol, Any


class LossFunction(Protocol):
    """Protocol class for neural network loss functions"""

    @staticmethod
    def calculate(y_target, y_pred) -> Any:
        ...

    @staticmethod
    def derivative(y_target, y_pred) -> Any:
        ...


class LogLoss:
    """Implements the log loss function for binary classification"""

    @staticmethod
    def calculate(y_target, y_pred) -> Any:
        return -1 * (y_target * np.log(y_pred) + (1 - y_target) * (np.log(1 - y_pred)))

    @staticmethod
    def derivative(y_target, y_pred) -> Any:
        return -1 * (y_target / y_pred) + ((1 - y_target) / (1 - y_pred))
