import cupy as cp

def weights_from_xavier(input_size: int, num_neurons: int) -> cp.ndarray:
    """Initialize weights using Xavier initialization for Tanh activations."""
    return cp.random.normal(
            0,
            cp.sqrt(2.0 / (input_size + num_neurons)),
            size=(input_size, num_neurons)
        )

def weights_from_he(input_size: int, num_neurons: int) -> cp.ndarray:
    """Initialize weights using He initialization for ReLU activations."""
    return cp.random.normal(
            0,
            cp.sqrt(2.0 / input_size),
            size=(input_size, num_neurons)
        )