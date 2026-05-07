import cupy as cp
from model.network import Network

class AdamWOptimizer:
    """
    AdamW optimizer (Loshchilov & Hutter, 2019).

    Decouples weight decay from the adaptive gradient update:

        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat   = m_t / (1 - beta1^t)
        v_hat   = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * lambda * theta_{t-1}

    Weight decay is applied only to weight matrices (apply_weight_decay=True),
    not to bias vectors, matching the convention used throughout this codebase.

    EmbeddingLayer returns an empty list from parameter_items() and falls back to
    its own update_parameters() for scatter-add embedding updates. DropoutLayer
    also returns an empty list; it inherits the no-op default from BaseLayer.
    """

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize AdamWOptimizer.

        Args:
            beta1: Exponential decay rate for the first moment estimate
            beta2: Exponential decay rate for the second moment estimate
            epsilon: Small constant for numerical stability
            weight_decay: Decoupled weight decay coefficient (lambda)
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self._first_moments: dict[tuple, cp.ndarray] = {}
        self._second_moments: dict[tuple, cp.ndarray] = {}
        self.step_count: int = 0

    def step(self, network: Network, learning_rate: float) -> None:
        """
        Apply one AdamW update step to all network parameters.

        Args:
            network: The Network whose parameters to update
            learning_rate: Current learning rate (supports cosine annealing schedules)
        """
        self.step_count += 1

        first_moment_bias_correction = 1.0 - self.beta1 ** self.step_count
        second_moment_bias_correction = 1.0 - self.beta2 ** self.step_count

        for layer_idx, layer in enumerate(network.layers):
            items = layer.parameter_items()

            if not items:
                layer.update_parameters(learning_rate=learning_rate, weight_decay_lambda=0.0, momentum=0.0)
                continue

            for name, param, grad, apply_wd in items:
                if grad is None:
                    continue

                key = (layer_idx, name)

                if key not in self._first_moments:
                    self._first_moments[key] = cp.zeros_like(param)
                    self._second_moments[key] = cp.zeros_like(param)

                first_moment = self._first_moments[key]
                second_moment = self._second_moments[key]

                first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * grad
                second_moment = self.beta2 * second_moment + (1.0 - self.beta2) * grad ** 2

                self._first_moments[key] = first_moment
                self._second_moments[key] = second_moment

                first_moment_corrected = first_moment / first_moment_bias_correction
                second_moment_corrected = second_moment / second_moment_bias_correction

                param -= learning_rate * first_moment_corrected / (cp.sqrt(second_moment_corrected) + self.epsilon)

                if apply_wd and self.weight_decay > 0.0:
                    param -= learning_rate * self.weight_decay * param

    def reset(self) -> None:
        """
        Clear all moment buffers and reset the step counter.

        Call this after loading a checkpoint to start with fresh Adam state.
        """
        self._first_moments = {}
        self._second_moments = {}
        self.step_count = 0
