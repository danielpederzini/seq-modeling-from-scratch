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

def softmax(input: cp.ndarray) -> cp.ndarray:
    """
    Apply Softmax activation function with numerical stability.
    
    Args:
        input: Input array of shape (batch_size, num_classes)
        
    Returns:
        Softmax output probabilities in range (0, 1) summing to 1 per sample
    """
    input_shifted: cp.ndarray = input - cp.max(input, axis=1, keepdims=True)
    exp_input: cp.ndarray = cp.exp(input_shifted)
    return exp_input / cp.sum(exp_input, axis=1, keepdims=True)