import cupy as cp
from typing import Dict, Any, Optional
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
        self._input_history: list[cp.ndarray] = []
    
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
        self._input_history.append(self._last_input)
        return softmax(input=linear_output)

    def reset(self) -> None:
        """Clear stored input history between sequences."""
        self._input_history = []

    def backward_sequence(self, output_errors: list[cp.ndarray], batch_size: int, clip_value: Optional[float] = None) -> list[cp.ndarray]:
        """
        Backward pass over a full sequence.

        Accumulates weight and bias gradients across all timesteps, then
        propagates per-timestep input errors to the previous layer.

        Args:
            output_errors: Per-timestep error gradients from the loss, one array
                per timestep in chronological order.
            batch_size: Batch size used for gradient averaging.
            clip_value: Maximum allowed L2 norm for the gradient. If None,
                clipping is disabled.

        Returns:
            Per-timestep error gradients w.r.t. the input, for the previous layer.
        """
        timesteps = len(output_errors)
        self._weights_grad = self.clip_grad(
            sum(input.T @ error for input, error in zip(self._input_history, output_errors)) / (batch_size * timesteps),
            clip_value=clip_value
        )
        self._biases_grad = self.clip_grad(
            sum(cp.mean(error, axis=0) for error in output_errors) / timesteps,
            clip_value=clip_value
        )
        return [error @ self.weights.T for error in output_errors]
