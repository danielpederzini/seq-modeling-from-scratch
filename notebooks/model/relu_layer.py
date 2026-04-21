import cupy as cp
from typing import Optional, Dict, Any
from .layer import Layer
from .layer_commons import weights_from_he

class ReluLayer(Layer):
    """
    Layer with ReLU activation function.
    
    Applies ReLU (Rectified Linear Unit) activation: f(x) = max(0, x)
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize ReLU layer.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        super().__init__(weights=weights, biases=biases)
        self.last_linear_output: Optional[cp.ndarray] = None
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "ReluLayer":
        """
        Create a ReluLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys
            
        Returns:
            Initialized ReluLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")
        
        weights: cp.ndarray = weights_from_he(input_size=input_size, num_neurons=num_neurons)
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return ReluLayer(weights=weights, biases=biases)

    def relu(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply ReLU activation function.
        
        Args:
            input: Input array
            
        Returns:
            Activated output with negative values set to 0
        """
        return cp.maximum(0, input)
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: linear transformation followed by ReLU activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Activated output array of shape (batch_size, num_neurons)
        """
        linear_output = super().forward(input=input)
        self.last_linear_output = linear_output
        return self.relu(input=linear_output)
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: ReLU gradient followed by linear layer gradient.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            
        Returns:
            Error gradient to propagate to previous layer
        """
        relu_grad = output_error * (self.last_linear_output > 0)
        input_error = super().backward(output_error=relu_grad, batch_size=batch_size)
        return input_error
