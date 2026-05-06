import cupy as cp
import math
import matplotlib.pyplot as plt
from collections import Counter

class DataLoader():
    """
    Loads, tokenizes, and exposes training and validation data for sequence modeling.

    Applies Byte-Pair Encoding (BPE) compression to the raw text, builds a vocabulary
    from the merged token set, and converts sequences to integer indices for use with
    embedding layers.
    """

    def __init__(self, training_path: str, validation_path: str, num_merges: int = 500) -> None:
        """
        Load and tokenize training and validation text files using BPE.

        Args:
            training_path: Path to the plain-text training file.
            validation_path: Path to the plain-text validation file.
            num_merges: Number of BPE merge operations to learn from the training data.
        """
        self._training_path = training_path
        self._validation_path = validation_path

        with open(self._training_path, "r", encoding="utf-8") as file:
            training_text = file.read()

        with open(self._validation_path, "r", encoding="utf-8") as file:
            validation_text = file.read()

        print(f"Training BPE with {num_merges} merges...")
        self._merges, train_tokens = self._train_bpe(list(training_text), num_merges)
        validation_tokens = self._apply_merges(list(validation_text))

        self._vocabulary = sorted(set(train_tokens) | set(validation_tokens))
        self._token_to_index = {token: index for index, token in enumerate(self._vocabulary)}

        self._x_train = train_tokens[:-1]
        self._y_train = train_tokens[1:]
        self._x_test = validation_tokens[:-1]
        self._y_test = validation_tokens[1:]

        print(f"Training data length: X={len(self._x_train)}, Y={len(self._y_train)}")
        print(f"Validation data length: X={len(self._x_test)}, Y={len(self._y_test)}")
        print(f"Vocabulary size: {len(self._vocabulary)}")

    def _train_bpe(self, chars: list, num_merges: int) -> tuple:
        """
        Learn BPE merge rules from an initial character sequence.

        Iteratively finds the most frequent adjacent token pair and merges it
        into a single token, repeating for ``num_merges`` iterations.

        Args:
            chars: Initial character-level token list.
            num_merges: Maximum number of merge operations to perform.

        Returns:
            Tuple of (merges dict, final token list) where merges maps
            each (left, right) pair to its merged token.
        """
        tokens = chars[:]
        merges = {}

        for index in range(num_merges):
            counts = Counter(zip(tokens[:-1], tokens[1:]))
            if not counts:
                break
            best = max(counts, key=counts.__getitem__)
            new_token = best[0] + best[1]
            merges[best] = new_token
            tokens = self._merge_pair(tokens, best, new_token)
            if (index + 1) % 100 == 0:
                print(f"  merge {index + 1}/{num_merges}: '{best[0]}'+'{best[1]}' -> '{new_token}'  (vocab={len(set(tokens))})")

        return merges, tokens

    def _merge_pair(self, tokens: list, pair: tuple, merged: str) -> list:
        """
        Replace all occurrences of an adjacent token pair with the merged token.

        Args:
            tokens: Current token sequence.
            pair: Tuple of (left_token, right_token) to merge.
            merged: Replacement token string.

        Returns:
            New token list with the pair replaced by ``merged`` everywhere.
        """
        result = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
                result.append(merged)
                index += 2
            else:
                result.append(tokens[index])
                index += 1
        return result

    def _apply_merges(self, chars: list) -> list:
        """
        Apply the learned BPE merge rules to an arbitrary character sequence.

        Args:
            chars: Initial character-level token list to encode.

        Returns:
            Token list after applying all learned merges in order.
        """
        tokens = chars[:]
        for pair, merged in self._merges.items():
            tokens = self._merge_pair(tokens, pair, merged)
        return tokens

    def encode(self, text: str) -> list:
        """Encode a string into a list of BPE tokens."""
        return self._apply_merges(list(text))

    def decode(self, tokens: list) -> str:
        """Decode a list of BPE tokens back into a string."""
        return "".join(tokens)

    def get_indices(self) -> tuple:
        """
        Convert all token sequences to integer index arrays on the GPU.

        Returns:
            4-tuple of CuPy int32 arrays: (x_train, y_train, x_test, y_test).
        """
        x_train_indices = cp.asarray([self._token_to_index[token] for token in self._x_train], dtype=cp.int32)
        y_train_indices = cp.asarray([self._token_to_index[token] for token in self._y_train], dtype=cp.int32)
        x_test_indices = cp.asarray([self._token_to_index[token] for token in self._x_test], dtype=cp.int32)
        y_test_indices = cp.asarray([self._token_to_index[token] for token in self._y_test], dtype=cp.int32)
        return x_train_indices, y_train_indices, x_test_indices, y_test_indices

    def get_vocabulary(self) -> list:
        """Return the sorted vocabulary list (index → token)."""
        return self._vocabulary

    def char_to_index(self, token: str) -> int:
        """Look up the integer index for a BPE token."""
        return self._token_to_index[token]

    def index_to_char(self, index: int) -> str:
        """Look up the BPE token for an integer index."""
        return self._vocabulary[index]

    def plot_vocabulary(self, num_columns: int = 16, fig_width: int = 28, font_size: int = 11, font_family: str = "monospace") -> None:
        """
        Display the full vocabulary as a grid using matplotlib.

        Special characters are replaced with visible stand-ins:
        spaces → ``·``, newlines → ``\\n``, tabs → ``\\t``.

        Args:
            num_columns: Number of token columns per row.
            fig_width: Width of the figure in inches.
            font_size: Font size for each token label.
            font_family: Font family used for rendering tokens.
        """
        rows = math.ceil(len(self._vocabulary) / num_columns)

        fig, ax = plt.subplots(figsize=(fig_width, max(2, rows * 0.7)))

        for index, token in enumerate(self._vocabulary):
            row, col = divmod(index, num_columns)
            display_token = token.replace(" ", "_").replace("\n", "\\n").replace("\t", "\\t")
            y = rows - row - 1
            ax.text(col, y, display_token, ha="center", va="center", fontsize=font_size, family=font_family)

        ax.set_xlim(-0.5, num_columns - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_xticks(range(num_columns))
        ax.set_yticks(range(rows))
        ax.grid(True, linestyle=":", alpha=0.3)
        plt.axis("off")
        plt.tight_layout()
        plt.show()