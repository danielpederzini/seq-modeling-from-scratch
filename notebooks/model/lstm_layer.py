import cupy as cp
from typing import Optional, Dict, Any
from .layer_commons import weights_from_xavier, sigmoid
from .layer import BaseLayer

class LSTMLayer(BaseLayer):
    """
    Long Short-Term Memory (LSTM) layer.

    Implements the standard LSTM equations:
        forget_gate = sigmoid(x_t @ forget_weights    + h_{t-1} @ forget_recurrent_weights    + forget_biases)
        input_gate  = sigmoid(x_t @ input_weights     + h_{t-1} @ input_recurrent_weights     + input_biases)
        output_gate = sigmoid(x_t @ output_weights    + h_{t-1} @ output_recurrent_weights    + output_biases)
        candidate   = tanh(x_t @ candidate_weights    + h_{t-1} @ candidate_recurrent_weights + candidate_biases)
        cell_t      = forget_gate * cell_{t-1} + input_gate * candidate
        h_t         = output_gate * tanh(cell_t)

    Each gate has its own independent weight matrices:
        {gate}_weights           : (input_size,  num_neurons)
        {gate}_recurrent_weights : (num_neurons, num_neurons)
        {gate}_biases            : (num_neurons,)
    """

    def __init__(
        self,
        forget_weights: cp.ndarray,
        forget_recurrent_weights: cp.ndarray,
        forget_biases: cp.ndarray,
        input_weights: cp.ndarray,
        input_recurrent_weights: cp.ndarray,
        input_biases: cp.ndarray,
        output_weights: cp.ndarray,
        output_recurrent_weights: cp.ndarray,
        output_biases: cp.ndarray,
        candidate_weights: cp.ndarray,
        candidate_recurrent_weights: cp.ndarray,
        candidate_biases: cp.ndarray,
    ) -> None:
        """
        Initialize LSTMLayer.

        Args:
            forget_weights: Forget gate input weights of shape (input_size, num_neurons)
            forget_recurrent_weights: Forget gate recurrent weights of shape (num_neurons, num_neurons)
            forget_biases: Forget gate biases of shape (num_neurons,)
            input_weights: Input gate input weights of shape (input_size, num_neurons)
            input_recurrent_weights: Input gate recurrent weights of shape (num_neurons, num_neurons)
            input_biases: Input gate biases of shape (num_neurons,)
            output_weights: Output gate input weights of shape (input_size, num_neurons)
            output_recurrent_weights: Output gate recurrent weights of shape (num_neurons, num_neurons)
            output_biases: Output gate biases of shape (num_neurons,)
            candidate_weights: Candidate cell input weights of shape (input_size, num_neurons)
            candidate_recurrent_weights: Candidate cell recurrent weights of shape (num_neurons, num_neurons)
            candidate_biases: Candidate cell biases of shape (num_neurons,)
        """
        self.forget_weights = forget_weights
        self.forget_recurrent_weights = forget_recurrent_weights
        self.forget_biases = forget_biases
        self.input_weights = input_weights
        self.input_recurrent_weights = input_recurrent_weights
        self.input_biases = input_biases
        self.output_weights = output_weights
        self.output_recurrent_weights = output_recurrent_weights
        self.output_biases = output_biases
        self.candidate_weights = candidate_weights
        self.candidate_recurrent_weights = candidate_recurrent_weights
        self.candidate_biases = candidate_biases

        self.hidden_size: int = forget_biases.shape[0]
        self.hidden_state: Optional[cp.ndarray] = None
        self.cell_state: Optional[cp.ndarray] = None

        self.forget_weights_grad: Optional[cp.ndarray] = None
        self.forget_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.forget_biases_grad: Optional[cp.ndarray] = None
        self.input_weights_grad: Optional[cp.ndarray] = None
        self.input_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.input_biases_grad: Optional[cp.ndarray] = None
        self.output_weights_grad: Optional[cp.ndarray] = None
        self.output_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.output_biases_grad: Optional[cp.ndarray] = None
        self.candidate_weights_grad: Optional[cp.ndarray] = None
        self.candidate_recurrent_weights_grad: Optional[cp.ndarray] = None
        self.candidate_biases_grad: Optional[cp.ndarray] = None

        self.forget_weights_velocity: Optional[cp.ndarray] = None
        self.forget_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.forget_biases_velocity: Optional[cp.ndarray] = None
        self.input_weights_velocity: Optional[cp.ndarray] = None
        self.input_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.input_biases_velocity: Optional[cp.ndarray] = None
        self.output_weights_velocity: Optional[cp.ndarray] = None
        self.output_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.output_biases_velocity: Optional[cp.ndarray] = None
        self.candidate_weights_velocity: Optional[cp.ndarray] = None
        self.candidate_recurrent_weights_velocity: Optional[cp.ndarray] = None
        self.candidate_biases_velocity: Optional[cp.ndarray] = None

        self.input_history: list[cp.ndarray] = []
        self.prev_hidden_history: list[cp.ndarray] = []
        self.prev_cell_history: list[cp.ndarray] = []
        self.forget_gate_history: list[cp.ndarray] = []
        self.input_gate_history: list[cp.ndarray] = []
        self.output_gate_history: list[cp.ndarray] = []
        self.candidate_history: list[cp.ndarray] = []
        self.cell_state_history: list[cp.ndarray] = []
        self.input_errors: list[cp.ndarray] = []

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "LSTMLayer":
        """
        Create an LSTMLayer instance from a definition dictionary.

        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys

        Returns:
            Initialized LSTMLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")

        return LSTMLayer(
            forget_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            forget_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            forget_biases=cp.zeros(shape=(num_neurons,)),
            input_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            input_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            input_biases=cp.zeros(shape=(num_neurons,)),
            output_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            output_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            output_biases=cp.zeros(shape=(num_neurons,)),
            candidate_weights=weights_from_xavier(input_size=input_size, num_neurons=num_neurons),
            candidate_recurrent_weights=weights_from_xavier(input_size=num_neurons, num_neurons=num_neurons),
            candidate_biases=cp.zeros(shape=(num_neurons,)),
        )

    def reset_state(self, batch_size: Optional[int] = None, dtype: cp.dtype = cp.float32) -> None:
        """
        Reset both recurrent states and sequence histories.

        Args:
            batch_size: If provided, initializes zero states with this batch size.
                If None, clears states and lets next forward initialize them lazily.
            dtype: Data type used when creating zero-initialized states.
        """
        if batch_size is None:
            self.hidden_state = None
            self.cell_state = None
        else:
            self.hidden_state = cp.zeros((batch_size, self.hidden_size), dtype=dtype)
            self.cell_state = cp.zeros((batch_size, self.hidden_size), dtype=dtype)

        self.input_history = []
        self.prev_hidden_history = []
        self.prev_cell_history = []
        self.forget_gate_history = []
        self.input_gate_history = []
        self.output_gate_history = []
        self.candidate_history = []
        self.cell_state_history = []
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
        if self.hidden_state is None or self.hidden_state.shape[0] != input.shape[0]:
            self.hidden_state = cp.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)
            self.cell_state = cp.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        prev_hidden = self.hidden_state
        prev_cell = self.cell_state

        forget_gate = sigmoid(input @ self.forget_weights + prev_hidden @ self.forget_recurrent_weights + self.forget_biases)
        input_gate = sigmoid(input @ self.input_weights + prev_hidden @ self.input_recurrent_weights + self.input_biases)
        output_gate = sigmoid(input @ self.output_weights + prev_hidden @ self.output_recurrent_weights + self.output_biases)
        candidate = cp.tanh(input @ self.candidate_weights + prev_hidden @ self.candidate_recurrent_weights + self.candidate_biases)

        new_cell = forget_gate * prev_cell + input_gate * candidate
        new_hidden = output_gate * cp.tanh(new_cell)

        self.hidden_state = new_hidden
        self.cell_state = new_cell

        self.input_history.append(input)
        self.prev_hidden_history.append(prev_hidden)
        self.prev_cell_history.append(prev_cell)
        self.forget_gate_history.append(forget_gate)
        self.input_gate_history.append(input_gate)
        self.output_gate_history.append(output_gate)
        self.candidate_history.append(candidate)
        self.cell_state_history.append(new_cell)

        return new_hidden

    def describe(self) -> str:
        """
        Get a formatted description of this layer.

        Returns:
            String description of the layer with shape and parameter information
        """
        layer_type: str = type(self).__name__
        weights_shape: tuple = self.forget_weights.shape
        recurrent_weights_shape: tuple = self.forget_recurrent_weights.shape
        layer_params: int = self.parameter_count()

        return f"{layer_type}\n  Weights Shape: {weights_shape} | Recurrent Weights Shape: {recurrent_weights_shape} | Gates: 4\n  Parameters: {layer_params:,}"

    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters, including recurrent weights.

        Returns:
            Total number of trainable parameters
        """
        return int(
            self.forget_weights.size + self.forget_recurrent_weights.size + self.forget_biases.size
            + self.input_weights.size + self.input_recurrent_weights.size + self.input_biases.size
            + self.output_weights.size + self.output_recurrent_weights.size + self.output_biases.size
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

        accumulated_forget_weights_grad = cp.zeros_like(self.forget_weights)
        accumulated_forget_recurrent_weights_grad = cp.zeros_like(self.forget_recurrent_weights)
        accumulated_forget_biases_grad = cp.zeros_like(self.forget_biases)
        accumulated_input_weights_grad = cp.zeros_like(self.input_weights)
        accumulated_input_recurrent_weights_grad = cp.zeros_like(self.input_recurrent_weights)
        accumulated_input_biases_grad = cp.zeros_like(self.input_biases)
        accumulated_output_weights_grad = cp.zeros_like(self.output_weights)
        accumulated_output_recurrent_weights_grad = cp.zeros_like(self.output_recurrent_weights)
        accumulated_output_biases_grad = cp.zeros_like(self.output_biases)
        accumulated_candidate_weights_grad = cp.zeros_like(self.candidate_weights)
        accumulated_candidate_recurrent_weights_grad = cp.zeros_like(self.candidate_recurrent_weights)
        accumulated_candidate_biases_grad = cp.zeros_like(self.candidate_biases)

        accumulated_hidden_error = cp.zeros_like(self.prev_hidden_history[0])
        accumulated_cell_error = cp.zeros_like(self.prev_cell_history[0])
        per_step_input_errors = []

        for input, prev_hidden, prev_cell, forget_gate, input_gate, output_gate, candidate, cell_state, direct_error in zip(
            reversed(self.input_history),
            reversed(self.prev_hidden_history),
            reversed(self.prev_cell_history),
            reversed(self.forget_gate_history),
            reversed(self.input_gate_history),
            reversed(self.output_gate_history),
            reversed(self.candidate_history),
            reversed(self.cell_state_history),
            reversed(output_errors),
        ):
            hidden_error = accumulated_hidden_error + direct_error
            tanh_cell = cp.tanh(cell_state)

            output_gate_preactivation_grad = hidden_error * tanh_cell * output_gate * (1.0 - output_gate)
            cell_error = hidden_error * output_gate * (1.0 - tanh_cell ** 2) + accumulated_cell_error

            forget_gate_preactivation_grad = cell_error * prev_cell * forget_gate * (1.0 - forget_gate)
            input_gate_preactivation_grad = cell_error * candidate * input_gate * (1.0 - input_gate)
            candidate_preactivation_grad = cell_error * input_gate * (1.0 - candidate ** 2)

            accumulated_forget_weights_grad += input.T @ forget_gate_preactivation_grad / batch_size
            accumulated_forget_recurrent_weights_grad += prev_hidden.T @ forget_gate_preactivation_grad / batch_size
            accumulated_forget_biases_grad += cp.mean(forget_gate_preactivation_grad, axis=0)

            accumulated_input_weights_grad += input.T @ input_gate_preactivation_grad / batch_size
            accumulated_input_recurrent_weights_grad += prev_hidden.T @ input_gate_preactivation_grad / batch_size
            accumulated_input_biases_grad += cp.mean(input_gate_preactivation_grad, axis=0)

            accumulated_output_weights_grad += input.T @ output_gate_preactivation_grad / batch_size
            accumulated_output_recurrent_weights_grad += prev_hidden.T @ output_gate_preactivation_grad / batch_size
            accumulated_output_biases_grad += cp.mean(output_gate_preactivation_grad, axis=0)

            accumulated_candidate_weights_grad += input.T @ candidate_preactivation_grad / batch_size
            accumulated_candidate_recurrent_weights_grad += prev_hidden.T @ candidate_preactivation_grad / batch_size
            accumulated_candidate_biases_grad += cp.mean(candidate_preactivation_grad, axis=0)

            accumulated_cell_error = cell_error * forget_gate
            accumulated_hidden_error = (
                forget_gate_preactivation_grad @ self.forget_recurrent_weights.T
                + input_gate_preactivation_grad @ self.input_recurrent_weights.T
                + output_gate_preactivation_grad @ self.output_recurrent_weights.T
                + candidate_preactivation_grad @ self.candidate_recurrent_weights.T
            )

            input_error = (
                forget_gate_preactivation_grad @ self.forget_weights.T
                + input_gate_preactivation_grad @ self.input_weights.T
                + output_gate_preactivation_grad @ self.output_weights.T
                + candidate_preactivation_grad @ self.candidate_weights.T
            )
            per_step_input_errors.append(input_error)

        self.forget_weights_grad = self.clip_grad(grad=accumulated_forget_weights_grad / timesteps)
        self.forget_recurrent_weights_grad = self.clip_grad(grad=accumulated_forget_recurrent_weights_grad / timesteps)
        self.forget_biases_grad = accumulated_forget_biases_grad / timesteps
        self.input_weights_grad = self.clip_grad(grad=accumulated_input_weights_grad / timesteps)
        self.input_recurrent_weights_grad = self.clip_grad(grad=accumulated_input_recurrent_weights_grad / timesteps)
        self.input_biases_grad = accumulated_input_biases_grad / timesteps
        self.output_weights_grad = self.clip_grad(grad=accumulated_output_weights_grad / timesteps)
        self.output_recurrent_weights_grad = self.clip_grad(grad=accumulated_output_recurrent_weights_grad / timesteps)
        self.output_biases_grad = accumulated_output_biases_grad / timesteps
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
            ("forget_weights", "forget_weights_grad", "forget_weights_velocity"),
            ("forget_recurrent_weights", "forget_recurrent_weights_grad", "forget_recurrent_weights_velocity"),
            ("input_weights", "input_weights_grad", "input_weights_velocity"),
            ("input_recurrent_weights", "input_recurrent_weights_grad", "input_recurrent_weights_velocity"),
            ("output_weights", "output_weights_grad", "output_weights_velocity"),
            ("output_recurrent_weights", "output_recurrent_weights_grad", "output_recurrent_weights_velocity"),
            ("candidate_weights", "candidate_weights_grad", "candidate_weights_velocity"),
            ("candidate_recurrent_weights", "candidate_recurrent_weights_grad", "candidate_recurrent_weights_velocity"),
        ]
        bias_params = [
            ("forget_biases", "forget_biases_grad", "forget_biases_velocity"),
            ("input_biases", "input_biases_grad", "input_biases_velocity"),
            ("output_biases", "output_biases_grad", "output_biases_velocity"),
            ("candidate_biases", "candidate_biases_grad", "candidate_biases_velocity"),
        ]

        for param_name, grad_name, vel_name in weight_params:
            grad = getattr(self, grad_name)
            if grad is None:
                continue
            param = getattr(self, param_name)
            if momentum > 0.0:
                vel = getattr(self, vel_name) if getattr(self, vel_name) is not None else cp.zeros_like(param)
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
                vel = getattr(self, vel_name) if getattr(self, vel_name) is not None else cp.zeros_like(param)
                vel = momentum * vel + grad
                setattr(self, vel_name, vel)
                setattr(self, param_name, param - learning_rate * vel)
            else:
                setattr(self, param_name, param - learning_rate * grad)
