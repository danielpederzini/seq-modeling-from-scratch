import cupy as cp
from typing import Dict, Any
from .layer import Layer
from .layer_commons import weights_from_he, softmax

class SoftmaxLayer(Layer):
    """
    Layer with Softmax activation function.
    
    Applies Softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j))
    Typically used for multi-class classification output layer.
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize Softmax layer.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        super().__init__(weights=weights, biases=biases)
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "SoftmaxLayer":
        """
        Create a SoftmaxLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys
            
        Returns:
            Initialized SoftmaxLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")
        
        weights: cp.ndarray = weights_from_he(input_size=input_size, num_neurons=num_neurons)
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return SoftmaxLayer(weights=weights, biases=biases)
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: linear transformation followed by Softmax activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Softmax output probabilities of shape (batch_size, num_neurons)
        """
        linear_output = super().forward(input=input)
        return softmax(input=linear_output)
