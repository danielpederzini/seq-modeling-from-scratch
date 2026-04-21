import cupy as cp
from typing import Dict, Any
from .layer import Layer

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
        
        weights: cp.ndarray = cp.random.normal(
            0,
            cp.sqrt(2.0 / input_size),
            size=(input_size, num_neurons)
        )
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return SoftmaxLayer(weights=weights, biases=biases)
    
    def softmax(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply Softmax activation function with numerical stability.
        
        Args:
            input: Input array of shape (batch_size, num_classes)
            
        Returns:
            Softmax output probabilities in range (0, 1) summing to 1 per sample
        """
        input_shifted: cp.ndarray = input - cp.max(input, axis=1, keepdims=True)
        exp_input: cp.ndarray = cp.exp(input_shifted)
        return exp_input / cp.sum(exp_input, axis=1, keepdims=True)
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: linear transformation followed by Softmax activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Softmax output probabilities of shape (batch_size, num_neurons)
        """
        linear_output = super().forward(input=input)
        return self.softmax(input=linear_output)
