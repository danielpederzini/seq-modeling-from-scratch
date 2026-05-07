import cupy as cp
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .layer_commons import weights_from_he

class BaseLayer(ABC):
    """
    Abstract base class defining the interface all network layers must implement.

    Network only depends on this contract — not on weights, biases, or any
    specific implementation detail.
    """

    @abstractmethod
    def forward(self, input: cp.ndarray) -> cp.ndarray: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def backward_sequence(self, output_errors: list, batch_size: int, clip_value: Optional[float] = None) -> list: ...

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0, momentum: float = 0.0) -> None: ...

    @abstractmethod
    def describe(self) -> str: ...

    @abstractmethod
    def parameter_count(self) -> int: ...

    def parameter_items(self) -> list[tuple]:
        """
        Return (name, param, grad, apply_weight_decay) tuples for all trainable parameters.

        Used by external optimizers (e.g. AdamW) to iterate over parameters without
        coupling to each layer's internal update logic. Returns an empty list by default;
        layers with trainable parameters must override this method.
        """
        return []


class Layer(BaseLayer):
    """
    Base dense (fully-connected) layer.

    Implements a shared linear transformation (input @ weights + biases).
    Subclasses extend this with activation functions.
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
        self._last_input: Optional[cp.ndarray] = None
        self._weights_grad: Optional[cp.ndarray] = None
        self._biases_grad: Optional[cp.ndarray] = None
    
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

    def parameter_items(self) -> list[tuple]:
        return [
            ("weights", self.weights, self._weights_grad, True),
            ("biases", self.biases, self._biases_grad, False),
        ]
        
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
        self._last_input = input
        dot_product = input @ self.weights
        return dot_product + self.biases
    
    def backward(self, output_error: cp.ndarray, batch_size: int, clip_value: Optional[float] = None) -> cp.ndarray:
        """
        Backward pass: compute gradients and propagate error.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            clip_value: Maximum allowed L2 norm for the gradient. If None,
                clipping is disabled.
            
        Returns:
            Error gradient to propagate to previous layer
        """
        weights_grad = self._last_input.T @ output_error / batch_size
        self._weights_grad = self.clip_grad(grad=weights_grad, clip_value=clip_value)
        self._biases_grad = self.clip_grad(grad=cp.mean(output_error, axis=0), clip_value=clip_value)
        
        return output_error @ self.weights.T

    def reset(self) -> None:
        """
        Reset any per-sequence state (e.g. input history). No-op for stateless layers.
        """
        pass

    def backward_sequence(self, output_errors: list, batch_size: int, clip_value: Optional[float] = None) -> list:
        """
        Backward pass over a full sequence.

        Default implementation for stateless layers: each timestep's gradient
        is computed independently. Subclasses that maintain recurrent state or
        accumulate history across timesteps must override this method.

        Args:
            output_errors: Per-timestep error gradients from the next layer
            batch_size: Batch size used for gradient averaging
            clip_value: Maximum allowed L2 norm for the gradient. If None,
                clipping is disabled.
        Returns:
            Per-timestep error gradients for the previous layer
        """
        return [self.backward(error, batch_size, clip_value=clip_value) for error in output_errors]
