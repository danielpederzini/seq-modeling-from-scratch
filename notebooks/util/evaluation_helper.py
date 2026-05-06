import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model.network import Network
from .data_loader import DataLoader

class EvaluationHelper:
    def __init__(self, model: Network, data_loader: DataLoader, vocabulary: list[str]) -> None:
        """
        Args:
            model: Trained network to evaluate.
            data_loader: DataLoader used to encode/decode tokens.
            vocabulary: Full vocabulary list mapping indices to tokens.
        """
        self.model = model
        self.data_loader = data_loader
        self.vocabulary = vocabulary

    def _format_token(self, token: str) -> str:
        return token.replace('\n', '\\n').replace(' ', '_').replace('\t', '\\t')

    def _get_embeddings(self) -> np.ndarray:
        return cp.asnumpy(self.model.layers[0].embeddings)

    def _top_indices_by_freq(self, x_test_indices: cp.ndarray, num_top: int) -> list[int]:
        freq_counter = Counter(int(token_index) for token_index in x_test_indices.get())
        return [token_index for token_index, _ in freq_counter.most_common(num_top)]

    def _index_labels(self, indices: list[int]) -> list[str]:
        return [self._format_token(self.data_loader.index_to_char(token_index)) for token_index in indices]

    def _length_colors(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        token_lengths = np.array([len(self.data_loader.index_to_char(token_index)) for token_index in indices])
        norm = (token_lengths - token_lengths.min()) / max(1, token_lengths.max() - token_lengths.min())
        return plt.cm.plasma(norm), token_lengths

    def _collect_hidden_states(self, sentence: str, layer_index: int) -> tuple[list[str], np.ndarray]:
        self.model.reset_states(batch_size=1, dtype=cp.float32)
        tokens = self.data_loader.encode(sentence)
        hidden_states = []
        for token in tokens:
            token_index = cp.asarray([self.data_loader.char_to_index(token)], dtype=cp.int32)
            outputs = self.model.forward(token_index)
            hidden_states.append(cp.asnumpy(outputs[layer_index][0]))
        return tokens, np.array(hidden_states)

    def generate_text(
        self,
        seed_text: str,
        length: int = 250,
        temperature: float = 0.8,
        print_during: bool = True,
    ) -> list[str]:
        """
        Autoregressively generate a token sequence from a seed string.

        Args:
            seed_text: Plain-text string used to prime the model's hidden state.
            length: Number of new tokens to generate after the seed.
            temperature: Sampling temperature. Higher values produce more
                random output; lower values make it more deterministic.
            print_during: If True, stream each generated token to stdout.

        Returns:
            List of BPE tokens comprising the seed and all generated tokens.
        """
        self.model.reset_states(batch_size=1, dtype=cp.float32)
        seed_tokens = self.data_loader.encode(seed_text)
        generated = list(seed_tokens)

        if print_during:
            print(seed_text, end="")

        for token in seed_tokens:
            token_index = cp.asarray([self.data_loader.char_to_index(token)], dtype=cp.int32)
            self.model.forward(token_index)

        current_token = seed_tokens[-1]

        for _ in range(length):
            token_index = cp.asarray([self.data_loader.char_to_index(current_token)], dtype=cp.int32)
            y_prob = self.model.forward(token_index)[-1][0]
            log_prob = cp.log(y_prob + 1e-9) / temperature
            scaled_prob = cp.exp(log_prob - cp.max(log_prob))
            scaled_prob /= cp.sum(scaled_prob)
            next_index = int(cp.random.choice(len(self.vocabulary), size=1, p=scaled_prob)[0].item())
            current_token = self.data_loader.index_to_char(next_index)
            generated.append(current_token)

            if print_during:
                print(current_token, end="", flush=True)

        return generated

    def plot_token_frequencies(
        self,
        x_test_indices: cp.ndarray,
        num_tokens: int = 5_000,
        num_top: int = 40,
        temperature: float = 0.8,
    ) -> None:
        """
        Plot a side-by-side bar chart comparing token frequencies between
        the real validation data and an equally-sized generated sample.

        Args:
            x_test_indices: Integer index array of validation tokens.
            num_tokens: Number of tokens to sample from both sources.
            num_top: How many of the most frequent real tokens to display.
            temperature: Sampling temperature used when generating text.
        """
        real_tokens = [self.data_loader.index_to_char(int(index)) for index in x_test_indices[:num_tokens].get()]
        generated_tokens = self.generate_text("HAMLET", length=num_tokens, temperature=temperature, print_during=False)

        real_counter = Counter(real_tokens)
        gen_counter = Counter(generated_tokens[:num_tokens])

        top_tokens = [token for token, _ in real_counter.most_common(num_top)]
        real_freqs = [real_counter[token] / num_tokens for token in top_tokens]
        gen_freqs = [gen_counter[token] / num_tokens for token in top_tokens]
        labels = [self._format_token(token) for token in top_tokens]

        x = cp.arange(num_top).get()
        width = 0.4

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(x - width / 2, real_freqs, width, label='Real (validation)', color='steelblue', alpha=0.85)
        ax.bar(x + width / 2, gen_freqs, width, label='Generated', color='coral', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
        ax.set_ylabel('Relative Frequency')
        ax.set_title(f'Token Frequency - Real vs Generated  (top {num_top} tokens, N={num_tokens:,})')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_embeddings_3d(self, x_test_indices: cp.ndarray, num_top: int = 50) -> None:
        """
        Visualise token embedding vectors in 3D space using PCA.

        Projects the full embedding matrix down to three principal components
        and plots the top-N most frequent validation tokens as arrows pointing
        from the origin. Tokens are coloured by their character length.

        Args:
            x_test_indices: Integer index array of validation tokens, used to
                determine token frequency for selecting the top-N tokens.
            num_top: Number of most-frequent tokens to display.
        """
        embeddings = self._get_embeddings()

        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        variance_percentages = pca.explained_variance_ratio_ * 100

        top_index_list = self._top_indices_by_freq(x_test_indices, num_top)
        coords = embeddings_3d[top_index_list]

        token_labels = self._index_labels(top_index_list)
        colors, token_lengths = self._length_colors(top_index_list)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for coord, label, color in zip(coords, token_labels, colors):
            ax.quiver(0, 0, 0, coord[0], coord[1], coord[2],
                      color=color, alpha=0.55, arrow_length_ratio=0.08, linewidth=1.0)
            ax.text(coord[0] * 1.1, coord[1] * 1.1, coord[2] * 1.1, label, fontsize=7, color=color)

        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=20, alpha=0.9, zorder=5)
        ax.set_xlabel(f'PC1 ({variance_percentages[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({variance_percentages[1]:.1f}%)')
        ax.set_zlabel(f'PC3 ({variance_percentages[2]:.1f}%)')
        ax.set_title(
            f'Token Embedding Vectors - 3D PCA  (top {num_top} tokens by frequency)\n'
            f'Colored by token length  |  Total variance explained: {sum(variance_percentages):.1f}%'
        )

        scalar_mappable = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=token_lengths.min(), vmax=token_lengths.max()))
        scalar_mappable.set_array([])
        plt.colorbar(scalar_mappable, ax=ax, shrink=0.45, pad=0.1, label='Token length (chars)')
        plt.tight_layout()
        plt.show()

    def plot_embeddings_tsne(self, x_test_indices: cp.ndarray, num_top: int = 100) -> None:
        """
        Project token embeddings into 2D space using t-SNE and plot a scatter.

        Args:
            x_test_indices: Integer index array of validation tokens, used to rank by frequency.
            num_top: Number of most-frequent tokens to project. Higher values reveal more
                structure but increase computation time.
        """
        embeddings = self._get_embeddings()
        top_index_list = self._top_indices_by_freq(x_test_indices, num_top)
        emb_subset = embeddings[top_index_list]

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_top - 1))
        emb_2d = tsne.fit_transform(emb_subset)

        labels = self._index_labels(top_index_list)
        colors, token_lengths = self._length_colors(top_index_list)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=60, alpha=0.85, zorder=5)
        for i, label in enumerate(labels):
            ax.annotate(label, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8, alpha=0.9,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE Token Embeddings  (top {num_top} tokens, colored by token length)')
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=token_lengths.min(), vmax=token_lengths.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Token length (chars)')
        plt.tight_layout()
        plt.show()

    def plot_hidden_state_trajectory(self, sentence: str, layer_index: int = 1) -> None:
        """
        Visualise how the hidden state evolves as the model reads a sentence,
        projected to 2D via PCA.

        Args:
            sentence: Plain-text sentence to run through the model.
            layer_index: Index of the layer whose hidden state to track. Defaults to 1
                (first recurrent layer).
        """
        tokens, hidden_matrix = self._collect_hidden_states(sentence, layer_index)
        pca = PCA(n_components=2)
        trajectory = pca.fit_transform(hidden_matrix)
        variance_percentages = pca.explained_variance_ratio_ * 100

        token_labels = [self._format_token(t) for t in tokens]
        token_count = len(tokens)
        point_colors = plt.cm.viridis(np.linspace(0, 1, token_count))

        fig, ax = plt.subplots(figsize=(12, 8))
        for token_pos in range(token_count - 1):
            ax.annotate('', xy=trajectory[token_pos + 1], xytext=trajectory[token_pos],
                        arrowprops=dict(arrowstyle='->', color=point_colors[token_pos], lw=1.5))
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=np.arange(token_count), cmap='viridis', s=80, zorder=5)
        for token_pos, label in enumerate(token_labels):
            ax.annotate(label, trajectory[token_pos], fontsize=9, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel(f'PC1 ({variance_percentages[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({variance_percentages[1]:.1f}%)')
        ax.set_title(
            f'Hidden State Trajectory - Layer {layer_index}  (dark -> light = start -> end)\n"{sentence[:60]}"'
        )
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, token_count - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Token position')
        plt.tight_layout()
        plt.show()

    def plot_temperature_sweep(
        self,
        seed_text: str = "HAMLET",
        temperatures: list[float] | None = None,
        length: int = 500,
    ) -> None:
        """
        Generate text at multiple temperatures and compare output diversity.

        Plots token entropy and unique-token count for each temperature, illustrating
        the creativity/coherence tradeoff.

        Args:
            seed_text: Seed string used for every generation run.
            temperatures: List of temperature values to sweep. Defaults to
                [0.2, 0.5, 0.8, 1.0, 1.2, 1.5].
            length: Number of tokens to generate per temperature.
        """
        if temperatures is None:
            temperatures = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
        entropies = []
        unique_counts = []

        for temp in temperatures:
            tokens = self.generate_text(seed_text, length=length, temperature=temp, print_during=False)
            counter = Counter(tokens)
            total = len(tokens)
            probs = np.array([count / total for count in counter.values()])
            entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
            entropies.append(entropy)
            unique_counts.append(len(counter))

        temp_labels = [str(t) for t in temperatures]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(temperatures, entropies, 'o-', color='steelblue', linewidth=2, markersize=8)
        for temperature, entropy in zip(temperatures, entropies):
            ax1.annotate(f'{entropy:.2f}', (temperature, entropy), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Token Entropy (nats)')
        ax1.set_title('Output Entropy vs Temperature')
        ax1.grid(True, alpha=0.3)

        ax2.bar(temp_labels, unique_counts, color='coral', alpha=0.85)
        for index, count in enumerate(unique_counts):
            ax2.text(index, count + 0.5, str(count), ha='center', fontsize=9)
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Unique Tokens Used')
        ax2.set_title(f'Vocabulary Diversity vs Temperature  (N={length:,})')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Temperature Sweep Analysis', fontsize=13)
        plt.tight_layout()
        plt.show()
