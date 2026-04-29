import cupy as cp
from .layer import BaseLayer

class DropoutLayer(BaseLayer):
    def __init__(self, rate: float = 0.2) -> None:
        self.rate = rate
        self.mask_history: list[cp.ndarray] = []
        self.training: bool = True

    @staticmethod
    def from_definition(definition):
        return DropoutLayer(rate=definition.get("rate", 0.2))

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        if not self.training:
            return input
        mask = (cp.random.random(input.shape, dtype=cp.float32) > self.rate).astype(cp.float32)
        mask /= (1.0 - self.rate)
        self.mask_history.append(mask)
        return input * mask

    def reset(self) -> None:
        self.mask_history = []

    def backward_sequence(self, output_errors, batch_size, clip_value=None):
        return [error * mask for error, mask in zip(output_errors, reversed(self.mask_history))]

    def update_parameters(self, learning_rate, weight_decay_lambda=0.0, momentum=0.0):
        pass

    def describe(self) -> str:
        return f"DropoutLayer\n  Rate: {self.rate}"

    def parameter_count(self) -> int:
        return 0