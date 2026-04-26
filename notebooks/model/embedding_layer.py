import cupy as cp
from typing import Dict, Any
from .layer import BaseLayer

class EmbeddingLayer(BaseLayer):
    """
    Embedding lookup layer.

    Maps integer token indices to dense embedding vectors and accumulates
    sparse gradient updates across a sequence during backpropagation.
    """

    def __init__(self, embeddings: cp.ndarray) -> None:
        """
        Args:
            embeddings: Weight matrix of shape (num_embeddings, embedding_dim).
        """
        self.embeddings = embeddings
        self.index_history: list[cp.ndarray] = []
        self.input_errors: list[cp.ndarray] = []

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "EmbeddingLayer":
        """
        Construct an EmbeddingLayer from a configuration dictionary.

        Expected keys: ``num_embeddings``, ``embedding_dim``, ``scale`` (optional, default 0.01).
        Embeddings are initialised with small random normal values scaled by ``scale``.
        """
        num_embeddings: int = definition["num_embeddings"]
        embedding_dim: int = definition["embedding_dim"]
        scale: float = definition.get("scale", 0.01)
        embeddings = cp.random.randn(num_embeddings, embedding_dim).astype(cp.float32) * scale
        return EmbeddingLayer(embeddings=embeddings)

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Look up embedding vectors for a batch of token indices.

        The indices are recorded in ``index_history`` for use during
        ``update_parameters``.

        Args:
            input: Integer array of shape (batch_size,) or (batch_size, T).

        Returns:
            Embedding vectors of shape (batch_size, embedding_dim) or
            (batch_size, T, embedding_dim).
        """
        self.index_history.append(input)
        return self.embeddings[input]

    def reset(self) -> None:
        """Clear index and gradient histories between sequences."""
        self.index_history = []
        self.input_errors = []

    def backward_sequence(self, output_errors: list[cp.ndarray], batch_size: int) -> list[cp.ndarray]:
        """
        Store per-step upstream gradients for use in ``update_parameters``.

        The embedding layer has no predecessor, so an empty list is returned.

        Args:
            output_errors: Per-step gradient arrays propagated from the next layer.
            batch_size: Number of sequences in the batch (unused here).

        Returns:
            Empty list — no further layers receive gradients from embeddings.
        """
        self.input_errors = output_errors
        return []

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0) -> None:
        """
        Apply sparse gradient updates to the embedding matrix.

        Uses ``cp.add.at`` for scatter-add so only the looked-up rows are updated.
        Weight decay is not applied to embeddings.

        Args:
            learning_rate: Step size for the gradient update.
            weight_decay_lambda: Unused; included for interface compatibility.
        """
        for indices, grad in zip(self.index_history, self.input_errors):
            cp.add.at(self.embeddings, indices, -learning_rate * grad)

    def describe(self) -> str:
        shape = self.embeddings.shape
        return f"EmbeddingLayer\n  Embeddings Shape: {shape}\n  Parameters: {self.parameter_count():,}"

    def parameter_count(self) -> int:
        return int(self.embeddings.shape[0] * self.embeddings.shape[1])
