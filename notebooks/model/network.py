import copy
import cupy as cp
from .layer import BaseLayer, Layer
from .embedding_layer import EmbeddingLayer
from .recurrent_layer import RecurrentLayer
from .gated_recurrent_layer import GatedRecurrentLayer
from .lstm_layer import LSTMLayer
from .softmax_layer import SoftmaxLayer

class Network:
    """
    Neural network composed of multiple layers.

    Orchestrates forward passes, backward passes, and parameter updates for
    networks built from dense and recurrent layers.
    """
    
    LAYER_TYPES: dict[str, type] = {
        "Embedding": EmbeddingLayer,
        "Recurrent": RecurrentLayer,
        "GatedRecurrent": GatedRecurrentLayer,
        "LSTM": LSTMLayer,
        "Softmax": SoftmaxLayer
    }
    
    def __init__(self, layer_definitions: list[dict]) -> None:
        """
        Initialize network with layer definitions.
        
        Args:
            layer_definitions: List of dictionaries defining each layer.
                For fully connected layers: 'type' (layer type), 'input_size', and 'num_neurons'.
        """
        self.layers = self.initialize_layers(layer_definitions=layer_definitions)
    
    def initialize_layers(self, layer_definitions: list[dict]) -> list[BaseLayer]:
        """
        Create layer instances based on definitions.
        
        Each layer type is responsible for creating itself from its definition.
        
        Args:
            layer_definitions: List of layer configuration dictionaries
            
        Returns:
            List of initialized BaseLayer objects
        """
        layers: list[BaseLayer] = []
        
        for definition in layer_definitions:
            layer_type: str = definition.get("type", "Layer")
            layer_class: type = self.LAYER_TYPES.get(layer_type, Layer)
            layers.append(layer_class.from_definition(definition))
        
        return layers

    def clone(self) -> "Network":
        """
        Create a full copy of the network state.

        Returns:
            Deep-copied network instance with independent parameters
        """
        return copy.deepcopy(self)
    
    def describe(self) -> None:
        """
        Print a formatted description of the network architecture.
        
        Delegates description to each layer.
        """
        description_lines: list[str] = [
            "=" * 80,
            "Network Architecture",
            "=" * 80,
        ]
        
        total_params: int = 0
        
        for layer_idx, layer in enumerate(self.layers, start=1):
            layer_desc: str = layer.describe()
            layer_params: int = layer.parameter_count()
            
            total_params += layer_params
            
            description_lines.append(f"\nLayer {layer_idx}: {layer_desc}")
        
        description_lines.append("\n" + "=" * 80)
        description_lines.append(f"Total Parameters: {total_params:,}")
        description_lines.append("=" * 80)
        
        print("\n".join(description_lines))

    def forward(self, input: cp.ndarray, print_shapes: bool = False) -> list[cp.ndarray]:
        """
        Forward pass for a single timestep through all layers.

        Recurrent layers use their internal state, which can be reset via
        ``reset_states`` between independent sequences.

        Args:
            input: Input tensor for the current timestep
            print_shapes: Whether to print the shape of the output at each layer

        Returns:
            List of output arrays from each layer
        """
        outputs: list[cp.ndarray] = []
        current_input: cp.ndarray = input

        for layer_index, layer in enumerate(self.layers):
            output = layer.forward(input=current_input)
            outputs.append(output)
            if print_shapes:
                print(f"Layer {layer_index} Output Shape: {output.shape}")
            current_input = output

        return outputs

    def reset_states(self, batch_size: int | None = None, dtype: cp.dtype = cp.float32) -> None:
        """
        Reset recurrent states and sequence histories for all layers.

        Args:
            batch_size: If provided, initialize zero states for recurrent layers.
                If None, clear states and lazily initialize on next forward.
            dtype: Data type used when initializing recurrent states.
        """
        for layer in self.layers:
            if isinstance(layer, (RecurrentLayer, GatedRecurrentLayer, LSTMLayer)):
                layer.reset_state(batch_size=batch_size, dtype=dtype)
            else:
                layer.reset()

    def backward_sequence(self, output_errors: list[cp.ndarray], batch_size: int) -> None:
        """
        Backward pass over a full sequence chunk (BPTT).

        Each layer computes and accumulates its own gradients across all timesteps.
        Call update_parameters afterwards to apply the accumulated gradients.

        Args:
            output_errors: Per-timestep error gradients from the loss, one array
                per timestep in chronological order
            batch_size: Batch size used for gradient averaging
        """
        current_errors = output_errors
        for layer in reversed(self.layers):
            current_errors = layer.backward_sequence(current_errors, batch_size)

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0, momentum: float = 0.0) -> None:
        """
        Update all layer parameters using computed gradients.
        
        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Regularization parameter for weight decay
            momentum: Momentum coefficient for SGD with momentum (0 disables it)
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate=learning_rate, weight_decay_lambda=weight_decay_lambda, momentum=momentum)

    def cce_loss(self, y_pred: cp.ndarray, y_true: cp.ndarray, epsilon=1e-15) -> cp.ndarray:
        """
        Compute categorical cross-entropy loss.

        Args:
            y_pred: Predicted class probabilities of shape (batch_size, num_classes)
            y_true: One-hot encoded target labels of shape (batch_size, num_classes)
            epsilon: Small value used to avoid taking the log of zero

        Returns:
            Scalar mean categorical cross-entropy loss
        """
        y_pred = cp.clip(y_pred, epsilon, 1. - epsilon)
        return -cp.mean(cp.sum(y_true * cp.log(y_pred), axis=1))