import cupy as cp
from typing import Optional, Dict, Any
from .layer_commons import weights_from_xavier, sigmoid
from .layer import BaseLayer

class GatedRecurrentLayer(BaseLayer):
    """
    Gated Recurrent Unit (GRU) layer.

    Implements the standard GRU equations:
        reset_gate  = sigmoid(x_t @ reset_weights  + h_{t-1} @ reset_recurrent_weights  + reset_biases)
        update_gate = sigmoid(x_t @ update_weights + h_{t-1} @ update_recurrent_weights + update_biases)
        candidate   = tanh(x_t @ candidate_weights + (reset_gate * h_{t-1}) @ candidate_recurrent_weights + candidate_biases)
        h_t         = (1 - update_gate) * h_{t-1} + update_gate * candidate

    Each gate has its own independent weight matrices:
        {gate}_weights           : (input_size,  num_neurons)
        {gate}_recurrent_weights : (num_neurons, num_neurons)
        {gate}_biases            : (num_neurons,)
    """

    def __init__(
        self,
        reset_weights: cp.ndarray,
        reset_recurrent_weights: cp.ndarray,
        reset_biases: cp.ndarray,
        update_weights: cp.ndarray,
        update_recurrent_weights: cp.ndarray,
        update_biases: cp.ndarray,
        candidate_weights: cp.ndarray,
        candidate_recurrent_weights: cp.ndarray,
        candidate_biases: cp.ndarray,
    ) -> None:
        """
        Initialize GatedRecurrentLayer.

        Args:
            reset_weights: Reset gate input weights of shape (input_size, num_neurons)
            reset_recurrent_weights: Reset gate recurrent weights of shape (num_neurons, num_neurons)
            reset_biases: Reset gate biases of shape (num_neurons,)
            update_weights: Update gate input weights of shape (input_size, num_neurons)
            update_recurrent_weights: Update gate recurrent weights of shape (num_neurons, num_neurons)
            update_biases: Update gate biases of shape (num_neurons,)
            candidate_weights: Candidate state input weights of shape (input_size, num_neurons)
            candidate_recurrent_weights: Candidate state recurrent weights of shape (num_neurons, num_neurons)
            candidate_biases: Candidate state biases of shape (num_neurons,)
        """
        self.reset_weights = reset_weights
        self.reset_recurrent_weights = reset_recurrent_weights
        self.reset_biases = reset_biases
        self.update_weights = update_weights
        self.update_recurrent_weights = update_recurrent_weights
        self.update_biases = update_biases
        self.candidate_weights = candidate_weights
        self.candidate_recurrent_weights = candidate_recurrent_weights
        self.candidate_biases = candidate_biases

        self.hidden_size: int = reset_biases.shape[0]
        self.state: Optional[cp.ndarray] = None

        self.reset_weights_grad: Optional[cp.ndarray] = None
        self.reset_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.reset_biases_grad: Optional[cp.ndarray] = None
        self.update_weights_grad: Optional[cp.ndarray] = None
        self.update_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.update_biases_grad: Optional[cp.ndarray] = None
        self.candidate_weights_grad: Optional[cp.ndarray] = None
        self.candidate_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.candidate_biases_grad: Optional[cp.ndarray] = None

        self.reset_weights_velocity: Optional[cp.ndarray] = None
        self.reset_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.reset_biases_velocity: Optional[cp.ndarray] = None
        self.update_weights_velocity: Optional[cp.ndarray] = None
        self.update_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.update_biases_velocity: Optional[cp.ndarray] = None
        self.candidate_weights_velocity: Optional[cp.ndarray] = None
        self.candidate_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.candidate_biases_velocity: Optional[cp.ndarray] = None

        self.input_history: list[cp.ndarray] = []
        self.prev_state_history: list[cp.ndarray] = []
        self.reset_gate_history: list[cp.ndarray] = []
        self.update_gate_history: list[cp.ndarray] = []
        self.candidate_history: list[cp.ndarray] = []
        self.input_errors: list[cp.ndarray] = []

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "GatedRecurrentLayer":
        """
        Create a GatedRecurrentLayer instance from a definition dictionary.

        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys

        Returns:
            Initialized GatedRecurrentLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")

        return GatedRecurrentLayer(
            reset_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            reset_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            reset_biases=cp.zeros(shape=(num_neurons,)),
            update_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            update_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            update_biases=cp.zeros(shape=(num_neurons,)),
            candidate_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            candidate_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            candidate_biases=cp.zeros(shape=(num_neurons,)),
        )

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
            self.state = cp.zeros((batch_size, self.hidden_size), dtype=dtype)

        self.input_history = []
        self.prev_state_history = []
        self.reset_gate_history = []
        self.update_gate_history = []
        self.candidate_history = []
        self.input_errors = []

    def reset(self) -> None:
        self.reset_state()

    def clip_grad(self, grad: cp.ndarray, clip_value: Optional[float] = None) -> cp.ndarray:
        if clip_value is None:
            return grad
        norm = cp.linalg.norm(grad)
        if norm > clip_value:
            grad = grad * (clip_value / norm)
        return grad

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass for a single timestep.

        Args:
            input: Input array of shape (batch_size, input_size)

        Returns:
            New hidden state of shape (batch_size, num_neurons)
        """
        if self.state is None or self.state.shape[0] != input.shape[0]:
            self.state = cp.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        prev_state = self.state

        reset_gate = sigmoid(input @ self.reset_weights + prev_state @ self.reset_recurrent_weights + self.reset_biases)
        update_gate = sigmoid(input @ self.update_weights + prev_state @ self.update_recurrent_weights + self.update_biases)
        candidate_preactivation = input @ self.candidate_weights + (reset_gate * prev_state) @ self.candidate_recurrent_weights + self.candidate_biases
        candidate = cp.tanh(candidate_preactivation)

        new_state = (1.0 - update_gate) * prev_state + update_gate * candidate
        self.state = new_state

        self.input_history.append(input)
        self.prev_state_history.append(prev_state)
        self.reset_gate_history.append(reset_gate)
        self.update_gate_history.append(update_gate)
        self.candidate_history.append(candidate)

        return new_state

    def describe(self) -> str:
        """
        Get a formatted description of this layer.

        Returns:
            String description of the layer with shape and parameter information
        """
        layer_type: str = type(self).__name__
        weights_shape: tuple = self.reset_weights.shape
        recurrent_weights_shape: tuple = self.reset_recurrent_weights.shape
        layer_params: int = self.parameter_count()

        return f"{layer_type}\n  Weights Shape: {weights_shape} | Recurrent Weights Shape: {recurrent_weights_shape} | Gates: 3\n  Parameters: {layer_params:,}"

    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters, including recurrent weights.

        Returns:
            Total number of trainable parameters
        """
        return int(
            self.reset_weights.size + self.reset_recurrent_weights.size + self.reset_biases.size
            + self.update_weights.size + self.update_recurrent_weights.size + self.update_biases.size
            + self.candidate_weights.size + self.candidate_recurrent_weights.size + self.candidate_biases.size
        )

    def backward_sequence(self, output_errors: list[cp.ndarray], batch_size: int) -> list[cp.ndarray]:
        """
        Backward pass over a full sequence chunk (BPTT).

        Accumulates gradients for all gate weights, recurrent weights, and biases
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

        accumulated_reset_weights_grad = cp.zeros_like(self.reset_weights)
        accumulated_reset_recurrent_weights_grad = cp.zeros_like(self.reset_recurrent_weights)
        accumulated_reset_biases_grad = cp.zeros_like(self.reset_biases)
        accumulated_update_weights_grad = cp.zeros_like(self.update_weights)
        accumulated_update_recurrent_weights_grad = cp.zeros_like(self.update_recurrent_weights)
        accumulated_update_biases_grad = cp.zeros_like(self.update_biases)
        accumulated_candidate_weights_grad = cp.zeros_like(self.candidate_weights)
        accumulated_candidate_recurrent_weights_grad = cp.zeros_like(self.candidate_recurrent_weights)
        accumulated_candidate_biases_grad = cp.zeros_like(self.candidate_biases)

        accumulated_state_error = cp.zeros_like(self.prev_state_history[0])
        per_step_input_errors = []

        for input, prev_state, reset_gate, update_gate, candidate, direct_error in zip(
            reversed(self.input_history),
            reversed(self.prev_state_history),
            reversed(self.reset_gate_history),
            reversed(self.update_gate_history),
            reversed(self.candidate_history),
            reversed(output_errors),
        ):
            state_error = accumulated_state_error + direct_error

            candidate_preactivation_grad = state_error * update_gate * (1.0 - candidate ** 2)
            update_gate_preactivation_grad = state_error * (candidate - prev_state) * update_gate * (1.0 - update_gate)
            reset_gate_hidden_grad = candidate_preactivation_grad @ self.candidate_recurrent_weights.T
            reset_gate_preactivation_grad = reset_gate_hidden_grad * prev_state * reset_gate * (1.0 - reset_gate)

            accumulated_reset_weights_grad += input.T @ reset_gate_preactivation_grad / batch_size
            accumulated_reset_recurrent_weights_grad += prev_state.T @ reset_gate_preactivation_grad / batch_size
            accumulated_reset_biases_grad += cp.mean(reset_gate_preactivation_grad, axis=0)

            accumulated_update_weights_grad += input.T @ update_gate_preactivation_grad / batch_size
            accumulated_update_recurrent_weights_grad += prev_state.T @ update_gate_preactivation_grad / batch_size
            accumulated_update_biases_grad += cp.mean(update_gate_preactivation_grad, axis=0)

            accumulated_candidate_weights_grad += input.T @ candidate_preactivation_grad / batch_size
            accumulated_candidate_recurrent_weights_grad += (reset_gate * prev_state).T @ candidate_preactivation_grad / batch_size
            accumulated_candidate_biases_grad += cp.mean(candidate_preactivation_grad, axis=0)

            accumulated_state_error = (
                state_error * (1.0 - update_gate)
                + reset_gate_preactivation_grad @ self.reset_recurrent_weights.T
                + update_gate_preactivation_grad @ self.update_recurrent_weights.T
                + reset_gate_hidden_grad * reset_gate
            )

            input_error = (
                reset_gate_preactivation_grad @ self.reset_weights.T
                + update_gate_preactivation_grad @ self.update_weights.T
                + candidate_preactivation_grad @ self.candidate_weights.T
            )
            per_step_input_errors.append(input_error)

        self.reset_weights_grad = self.clip_grad(grad=accumulated_reset_weights_grad / timesteps)
        self.reset_recurrent_weights_grad = self.clip_grad(grad=accumulated_reset_recurrent_weights_grad / timesteps)
        self.reset_biases_grad = accumulated_reset_biases_grad / timesteps
        self.update_weights_grad = self.clip_grad(grad=accumulated_update_weights_grad / timesteps)
        self.update_recurrent_weights_grad = self.clip_grad(grad=accumulated_update_recurrent_weights_grad / timesteps)
        self.update_biases_grad = accumulated_update_biases_grad / timesteps
        self.candidate_weights_grad = self.clip_grad(grad=accumulated_candidate_weights_grad / timesteps)
        self.candidate_recurrent_weights_grad = self.clip_grad(grad=accumulated_candidate_recurrent_weights_grad / timesteps)
        self.candidate_biases_grad = accumulated_candidate_biases_grad / timesteps
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
        weight_params = [
            ("reset_weights", "reset_weights_grad", "reset_weights_velocity"),
            ("reset_recurrent_weights", "reset_recurrent_weights_grad", "reset_recurrent_weights_velocity"),
            ("update_weights", "update_weights_grad", "update_weights_velocity"),
            ("update_recurrent_weights", "update_recurrent_weights_grad", "update_recurrent_weights_velocity"),
            ("candidate_weights", "candidate_weights_grad", "candidate_weights_velocity"),
            ("candidate_recurrent_weights", "candidate_recurrent_weights_grad", "candidate_recurrent_weights_velocity"),
        ]
        bias_params = [
            ("reset_biases", "reset_biases_grad", "reset_biases_velocity"),
            ("update_biases", "update_biases_grad", "update_biases_velocity"),
            ("candidate_biases", "candidate_biases_grad", "candidate_biases_velocity"),
        ]

        for param_name, grad_name, vel_name in weight_params:
            grad = getattr(self, grad_name)
            if grad is None:
                continue
            param = getattr(self, param_name)
            if momentum > 0.0:
                vel = getattr(self, vel_name) or cp.zeros_like(param)
                vel = momentum * vel + grad + weight_decay_lambda * param
                setattr(self, vel_name, vel)
                setattr(self, param_name, param - learning_rate * vel)
            else:
                setattr(self, param_name, param - learning_rate * (grad + weight_decay_lambda * param))

        for param_name, grad_name, vel_name in bias_params:
            grad = getattr(self, grad_name)
            if grad is None:
                continue
            param = getattr(self, param_name)
            if momentum > 0.0:
                vel = getattr(self, vel_name) or cp.zeros_like(param)
                vel = momentum * vel + grad
                setattr(self, vel_name, vel)
                setattr(self, param_name, param - learning_rate * vel)
            else:
                setattr(self, param_name, param - learning_rate * grad)
