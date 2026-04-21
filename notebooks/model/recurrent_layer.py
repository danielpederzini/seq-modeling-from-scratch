import cupy as cp
from typing import Optional, Dict, Any
from .layer_commons import weights_from_xavier
from .layer import Layer

class RecurrentLayer(Layer):
    """
    Layer with Tanh activation function.
    
    Applies Tanh activation: f(x) = tanh(x)
    """
    
    def __init__(self, weights: cp.ndarray, state_weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize Recurrent layer.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            state_weights: Weight matrix for the recurrent state of shape (num_neurons, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        super().__init__(weights=weights, biases=biases)
        self.state_weights = state_weights
        self.state: Optional[cp.ndarray] = None
        self.last_linear_output: Optional[cp.ndarray] = None
        self.last_prev_state: Optional[cp.ndarray] = None
        self.s_grad: Optional[cp.ndarray] = None
        self.state_error: Optional[cp.ndarray] = None
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "RecurrentLayer":
        """
        Create a RecurrentLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys
            
        Returns:
            Initialized RecurrentLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")
        
        weights: cp.ndarray = weights_from_xavier(input_size=input_size, num_neurons=num_neurons)
        state_weights: cp.ndarray = weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons)
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return RecurrentLayer(weights=weights, state_weights=state_weights, biases=biases)

    def tanh(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply Tanh activation function.
        
        Args:
            input: Input array
            
        Returns:
            Activated output with values in range (-1, 1)
        """
        return cp.tanh(input)

    def reset_state(self, batch_size: Optional[int] = None, dtype: cp.dtype = cp.float32) -> None:
        """
        Reset the recurrent hidden state.

        Args:
            batch_size: If provided, initializes a zero state with this batch size.
                If None, clears state and lets next forward initialize it lazily.
            dtype: Data type used when creating a zero-initialized state.
        """
        if batch_size is None:
            self.state = None
            return

        self.state = cp.zeros((batch_size, self.biases.shape[0]), dtype=dtype)
    
    def forward(self, input: cp.ndarray, prev_state: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Forward pass: linear transformation followed by Tanh activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            prev_state: Previous hidden state of shape (batch_size, num_neurons)
            
        Returns:
            Activated output array of shape (batch_size, num_neurons)
        """
        if prev_state is not None:
            self.state = prev_state

        if self.state is None or self.state.shape[0] != input.shape[0]:
            self.state = cp.zeros((input.shape[0], self.biases.shape[0]), dtype=input.dtype)

        linear_output = super().forward(input=input)
        linear_output += self.state @ self.state_weights

        self.last_prev_state = self.state
        self.last_linear_output = linear_output
        output_state = self.tanh(input=linear_output)
        self.state = output_state
        return output_state

    def describe(self) -> str:
        """
        Get a formatted description of this layer.
        
        Returns:
            String description of the layer with shape and parameter information
        """
        layer_type: str = type(self).__name__
        weights_shape: tuple = self.weights.shape
        biases_shape: tuple = self.biases.shape
        state_weights_shape: tuple = self.state_weights.shape
        layer_params: int = self.parameter_count()
        
        return f"{layer_type}\n  Weights Shape: {weights_shape} | State Weights Shape: {state_weights_shape} | Biases Shape: {biases_shape}\n  Parameters: {layer_params:,}"
    
    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters, including recurrent weights.

        Returns:
            Total number of trainable parameters
        """
        return super().parameter_count() + int(self.state_weights.shape[0] * self.state_weights.shape[1])
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: Tanh gradient followed by linear layer gradient.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            
        Returns:
            Error gradient to propagate to previous layer
        """
        tanh_output = self.tanh(input=self.last_linear_output)
        tanh_grad = output_error * (1 - tanh_output ** 2)

        if self.last_prev_state is not None:
            state_grad = self.last_prev_state.T @ tanh_grad / batch_size
            self.s_grad = self.clip_grad(grad=state_grad)
            self.state_error = tanh_grad @ self.state_weights.T
        else:
            self.s_grad = None
            self.state_error = None

        input_error = super().backward(output_error=tanh_grad, batch_size=batch_size)
        return input_error

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0) -> None:
        """
        Update this layer's trainable parameters, including recurrent weights.

        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Regularization parameter for weight decay
        """
        super().update_parameters(
            learning_rate=learning_rate,
            weight_decay_lambda=weight_decay_lambda,
        )

        if self.s_grad is not None:
            self.state_weights -= learning_rate * (
                self.s_grad + weight_decay_lambda * self.state_weights
            )
