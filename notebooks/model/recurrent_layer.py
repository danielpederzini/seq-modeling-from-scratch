import cupy as cp
from typing import Optional, Dict, Any
from .layer_commons import weights_from_xavier
from .layer import Layer

class RecurrentLayer(Layer):
    """
    Vanilla recurrent layer with Tanh activation.

    Implements the standard RNN equations:
        h_t = tanh(x_t @ weights + h_{t-1} @ state_weights + biases)

    Weight layout:
        weights       : (input_size,  num_neurons)
        state_weights : (num_neurons, num_neurons)
        biases        : (num_neurons,)
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
        self.s_grad: Optional[cp.ndarray] = None
        self.s_velocity: Optional[cp.ndarray] = None
        self.input_errors: list[cp.ndarray] = []
        self.input_history: list[cp.ndarray] = []
        self.prev_state_history: list[cp.ndarray] = []
        self.state_history: list[cp.ndarray] = []
    
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

    def reset_state(self, batch_size: Optional[int] = None, dtype: cp.dtype = cp.float32) -> None:
        """
        Reset the recurrent hidden state and sequence histories.

        Args:
            batch_size: If provided, initializes a zero state with this batch size.
                If None, clears state and lets next forward initialize it lazily.
            dtype: Data type used when creating a zero-initialized state.
        """
        if batch_size is None:
            self.state = None
        else:
            self.state = cp.zeros((batch_size, self.biases.shape[0]), dtype=dtype)

        self.input_history = []
        self.prev_state_history = []
        self.state_history = []        
        self.input_errors = []

    def reset(self) -> None:
        self.reset_state()

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass for a single timestep.

        Args:
            input: Input array of shape (batch_size, input_size)

        Returns:
            New hidden state of shape (batch_size, num_neurons)
        """
        if self.state is None or self.state.shape[0] != input.shape[0]:
            self.state = cp.zeros((input.shape[0], self.biases.shape[0]), dtype=input.dtype)

        prev_state = self.state
        linear_output = super().forward(input=input)
        linear_output += prev_state @ self.state_weights

        output_state = cp.tanh(linear_output)
        self.state = output_state

        self.input_history.append(self.last_input)
        self.prev_state_history.append(prev_state)
        self.state_history.append(output_state)

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
    
    def backward_sequence(self, output_errors: list[cp.ndarray], batch_size: int) -> list[cp.ndarray]:
        """
        Backward pass over a full sequence chunk (BPTT).

        Accumulates gradients for weights, state_weights, and biases
        across all timesteps using reverse-time unrolling. Each timestep's
        direct error is injected at the correct point.

        Args:
            output_errors: Per-timestep error gradients from the next layer,
                in chronological order
            batch_size: Batch size used for gradient averaging

        Returns:
            Per-timestep error gradients w.r.t. the input, for the previous layer
        """
        timesteps = len(output_errors)
        accumulated_w_grad = cp.zeros_like(self.weights)
        accumulated_s_grad = cp.zeros_like(self.state_weights)
        accumulated_b_grad = cp.zeros_like(self.biases)

        accumulated_state_error = cp.zeros_like(self.state_history[0])
        input_error = cp.zeros_like(self.input_history[0])
        per_step_input_errors = []

        for input, prev_state, state, direct_error in zip(
            reversed(self.input_history),
            reversed(self.prev_state_history),
            reversed(self.state_history),
            reversed(output_errors),
        ):
            accumulated_state_error += direct_error
            tanh_grad = accumulated_state_error * (1 - state ** 2)

            accumulated_w_grad += input.T @ tanh_grad / batch_size
            accumulated_s_grad += prev_state.T @ tanh_grad / batch_size
            accumulated_b_grad += cp.mean(tanh_grad, axis=0)

            input_error = tanh_grad @ self.weights.T
            per_step_input_errors.append(input_error)
            accumulated_state_error = tanh_grad @ self.state_weights.T

        self.w_grad = self.clip_grad(grad=accumulated_w_grad / timesteps)
        self.s_grad = self.clip_grad(grad=accumulated_s_grad / timesteps)
        self.b_grad = accumulated_b_grad / timesteps
        self.input_errors = list(reversed(per_step_input_errors))

        return self.input_errors

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0, momentum: float = 0.0) -> None:
        """
        Update this layer's trainable parameters, including recurrent weights.

        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Regularization parameter for weight decay
            momentum: Momentum coefficient passed through to all parameter updates
        """
        super().update_parameters(
            learning_rate=learning_rate,
            weight_decay_lambda=weight_decay_lambda,
            momentum=momentum,
        )

        if self.s_grad is not None:
            if momentum > 0.0:
                if self.s_velocity is None:
                    self.s_velocity = cp.zeros_like(self.state_weights)
                self.s_velocity = momentum * self.s_velocity + self.s_grad + weight_decay_lambda * self.state_weights
                self.state_weights -= learning_rate * self.s_velocity
            else:
                self.state_weights -= learning_rate * (
                    self.s_grad + weight_decay_lambda * self.state_weights
                )
