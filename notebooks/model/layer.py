import cupy as cp
from typing import Optional, Dict, Any
from .layer_commons import weights_from_he

class Layer:
    """
    Base fully connected layer for the neural network.

    Implements the shared linear transformation, gradient computation, and
    parameter update logic used by dense layers.
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize the layer with weights and biases.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        self.weights = weights
        self.biases = biases
        self.last_input: Optional[cp.ndarray] = None
        self.w_grad: Optional[cp.ndarray] = None
        self.b_grad: Optional[cp.ndarray] = None
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "Layer":
        """
        Create a Layer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys
            
        Returns:
            Initialized Layer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")
        
        weights: cp.ndarray = weights_from_he(input_size=input_size, num_neurons=num_neurons)
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return Layer(weights=weights, biases=biases)
    
    def describe(self) -> str:
        """
        Get a formatted description of this layer.
        
        Returns:
            String description of the layer with shape and parameter information
        """
        layer_type: str = type(self).__name__
        weights_shape: tuple = self.weights.shape
        biases_shape: tuple = self.biases.shape
        layer_params: int = self.parameter_count()
        
        return f"{layer_type}\n  Weights Shape: {weights_shape} | Biases Shape: {biases_shape}\n  Parameters: {layer_params:,}"

    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return int(self.weights.shape[0] * self.weights.shape[1] + self.biases.shape[0])
        
    def clip_grad(self, grad: cp.ndarray, clip_value: Optional[float] = None) -> cp.ndarray:
        """
        Optionally clip gradient using L2 norm.
        
        Args:
            grad: Gradient array to clip
            clip_value: Maximum allowed L2 norm for the gradient. If None,
                clipping is disabled.
            
        Returns:
            Gradient array, clipped only when a clip value is provided
        """
        if clip_value is None:
            return grad

        norm = cp.linalg.norm(grad)
        if norm > clip_value:
            grad = grad * (clip_value / norm)
            
        return grad
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: compute linear transformation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array of shape (batch_size, num_neurons)
        """
        self.last_input = input
        dot_product = input @ self.weights
        return dot_product + self.biases
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: compute gradients and propagate error.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            
        Returns:
            Error gradient to propagate to previous layer
        """
        w_grad = self.last_input.T @ output_error / batch_size
        self.w_grad = self.clip_grad(grad=w_grad)
        self.b_grad = self.clip_grad(grad=cp.mean(output_error, axis=0))
        
        return output_error @ self.weights.T

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0) -> None:
        """
        Update this layer's trainable parameters using the stored gradients.

        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Regularization parameter for weight decay
        """
        if self.w_grad is not None:
            self.weights -= learning_rate * (self.w_grad + weight_decay_lambda * self.weights)
        if self.b_grad is not None:
            self.biases -= self.b_grad * learning_rate
