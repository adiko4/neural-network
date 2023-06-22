import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .activations import ActivationFunction, ReLUActivation, SigmoidActivation
from .loss_function import LossFunction


class ActivationFunctionType(Enum):
    SIGMOID = SigmoidActivation
    RELU = ReLUActivation


@dataclass(frozen=True)
class NeuralNetworkLayerInfo:
    nodes_number: int
    activation_function_type: Optional[ActivationFunctionType]


@dataclass
class NeuralNetworkLayer:
    weights: np.ndarray
    biases: np.ndarray
    activation_function: ActivationFunction


@dataclass
class NeuralNetworkLayerGradients:
    dW: np.ndarray
    db: np.ndarray


@dataclass
class NeuralNetworkLayerCacheUnit:
    previous_activations: np.ndarray
    Z: np.ndarray


class ArtificialNeuralNetwork:
    """Artificial neural network with batch gradient descent learning algorithm,
    flexible to different activation functions and structure"""

    COST_PRINT_INTERVAL = 100

    def __init__(self, layers_info: List[NeuralNetworkLayerInfo], loss_function: LossFunction):
        self._layers: List[NeuralNetworkLayer] = ArtificialNeuralNetwork.construct_layers(layers_info)
        self._cache: List[NeuralNetworkLayerCacheUnit] = []
        self._loss_function = loss_function

    def predict(self, X):
        """Invokes model prediction using the trained weights and biases"""
        return self._forward_propagate(X)

    def fit(self, X, Y, learning_rate, iterations) -> List[int]:
        """Trains the model using batch gradient descent over the dataset"""
        cost_history = list()
        m = X.shape[1]

        for i in range(iterations):
            AL = self._forward_propagate(X, store_cache=True)

            cost = np.sum(self._loss_function.calculate(Y, AL)) / m
            if i % ArtificialNeuralNetwork.COST_PRINT_INTERVAL == 0:
                print(f"Iteration number {i} - Cost: {cost}")
            cost_history.append(cost)

            gradients = self._back_propagate(Y, AL)
            self._adjust_parameters(learning_rate, gradients)

            self._cache.clear()

        return cost_history

    def _forward_propagate(self, X, store_cache=False) -> np.ndarray:
        """Progpagates the inputs through the neural network and produces the predicted output"""
        activations = X

        for layer in self._layers:
            Z = np.dot(layer.weights, activations) + layer.biases
            if store_cache:
                # Saving previous activations and current z values
                self._cache.append(NeuralNetworkLayerCacheUnit(activations, Z))
            activations = layer.activation_function.calculate(Z)

        return activations

    def _back_propagate(self, Y, AL) -> List[NeuralNetworkLayerGradients]:
        """Solving for all layers weights and biases gradients"""
        m = AL.shape[1]
        gradients = []

        dA = self._loss_function.derivative(Y, AL)
        
        for current_layer, current_layer_cache in zip(reversed(self._layers), reversed(self._cache)):
            # Computing derivatives for the current layer
            dZ = dA * current_layer.activation_function.derivative(current_layer_cache.Z)
            dW = np.dot(dZ, current_layer_cache.previous_activations.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            # Inserting weights and biases gradients backwards for convinient adjustments later
            gradients.insert(0, NeuralNetworkLayerGradients(dW, db))
            
            # Computing derivatives of activations one layer backwards
            dA = np.dot(current_layer.weights.T, dZ)

        return gradients
    
    def _adjust_parameters(self, learning_rate, gradients) -> None:
        """Adjusting the weights and biases of each layer for minimizing cost function"""
        for layer, layer_gradients in zip(self._layers, gradients):
            layer.weights -= learning_rate * layer_gradients.dW
            layer.biases -= learning_rate * layer_gradients.db

    @staticmethod
    def construct_layers(layers_info: List[NeuralNetworkLayerInfo]) -> List[NeuralNetworkLayer]:
        """Creates a list of all layers with initialized weights and biases"""
        layers = []

        for i in range(len(layers_info) - 1):
            # TODO: Understand thorougly the He / Xavier initialization and add the right nominator per activation function and not a magic number
            weights_matrix = np.random.randn(layers_info[i + 1].nodes_number, layers_info[i].nodes_number) * np.sqrt(2 / layers_info[i].nodes_number)
            biases_matrix = np.zeros((layers_info[i + 1].nodes_number, 1))

            # Activation function type should be None, only at the first layer
            assert layers_info[i + 1].activation_function_type is not None

            layers.append(NeuralNetworkLayer(weights_matrix, biases_matrix, layers_info[i + 1].activation_function_type.value))
        
        return layers
