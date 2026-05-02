import cupy as cp
from typing import Optional
from .layer import BaseLayer

class DropoutLayer(BaseLayer):
    """
    Inverted dropout layer.

    Randomly zeroes activations during training with probability ``rate`` and
    rescales the remaining activations by ``1 / (1 - rate)`` so that expected
    magnitudes are preserved. During evaluation the layer is a no-op.
    """

    def __init__(self, rate: float = 0.2) -> None:
        """
        Args:
            rate: Fraction of activations to drop. Must be in [0, 1).
        """
        self.rate = rate
        self.training: bool = True
        self._mask_history: list[cp.ndarray] = []

    @staticmethod
    def from_definition(definition) -> "DropoutLayer":
        """
        Construct a DropoutLayer from a configuration dictionary.

        Expected key: ``rate`` (default 0.2).
        """
        return DropoutLayer(rate=definition.get("rate", 0.2))

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply dropout mask during training; pass through unchanged during evaluation.

        Args:
            input: Input activations of shape (batch_size, features).

        Returns:
            Masked and rescaled activations of the same shape.
        """
        if not self.training:
            return input
        mask = (cp.random.random(input.shape, dtype=cp.float32) > self.rate).astype(cp.float32)
        mask /= (1.0 - self.rate)
        self._mask_history.append(mask)
        return input * mask

    def reset(self) -> None:
        """Clear the stored dropout masks between sequences."""
        self._mask_history = []

    def backward_sequence(self, output_errors: list, batch_size: int, clip_value: Optional[float] = None) -> list:
        """
        Propagate gradients through the dropout masks.

        Args:
            output_errors: Per-timestep error gradients from the next layer.
            batch_size: Batch size (unused; present for interface compatibility).
            clip_value: Unused; present for interface compatibility.

        Returns:
            Per-timestep error gradients scaled by the saved dropout masks.
        """
        return [error * mask for error, mask in zip(output_errors, reversed(self._mask_history))]

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0, momentum: float = 0.0) -> None:
        """No-op: dropout has no trainable parameters."""
        pass

    def describe(self) -> str:
        """Return a formatted description of the dropout rate."""
        return f"DropoutLayer\n  Rate: {self.rate}"

    def parameter_count(self) -> int:
        """Return 0 — dropout has no trainable parameters."""
        return 0