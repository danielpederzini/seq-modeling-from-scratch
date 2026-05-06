import cupy as cp
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
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
        real_tokens = [self.data_loader.index_to_char(int(i)) for i in x_test_indices[:num_tokens].get()]
        generated_tokens = self.generate_text("HAMLET", length=num_tokens, temperature=temperature, print_during=False)

        real_counter = Counter(real_tokens)
        gen_counter = Counter(generated_tokens[:num_tokens])

        top_tokens = [token for token, _ in real_counter.most_common(num_top)]
        real_freqs = [real_counter[token] / num_tokens for token in top_tokens]
        gen_freqs = [gen_counter[token] / num_tokens for token in top_tokens]
        labels = [token.replace('\n', '↵').replace(' ', '·').replace('\t', '→') for token in top_tokens]

        x = cp.arange(num_top).get()
        width = 0.4

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(x - width / 2, real_freqs, width, label='Real (validation)', color='steelblue', alpha=0.85)
        ax.bar(x + width / 2, gen_freqs,  width, label='Generated',         color='coral',    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
        ax.set_ylabel('Relative Frequency')
        ax.set_title(f'Token Frequency – Real vs Generated  (top {num_top} tokens, N={num_tokens:,})')
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
        embeddings = cp.asnumpy(self.model.layers[0].embeddings)

        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        var_pct = pca.explained_variance_ratio_ * 100

        freq_counter = Counter(int(i) for i in x_test_indices.get())
        top_index_list = [index for index, _ in freq_counter.most_common(num_top)]
        coords = embeddings_3d[top_index_list]

        token_labels = [self.data_loader.index_to_char(index).replace('\n', '↵').replace(' ', '·') for index in top_index_list]
        top_indices_length = cp.array([len(self.data_loader.index_to_char(index)) for index in top_index_list])
        normalized_length = (top_indices_length - top_indices_length.min()) / max(1, top_indices_length.max() - top_indices_length.min())
        colors = plt.cm.plasma(normalized_length.get())

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for coord, label, color in zip(coords, token_labels, colors):
            ax.quiver(0, 0, 0, coord[0], coord[1], coord[2],
                      color=color, alpha=0.55, arrow_length_ratio=0.08, linewidth=1.0)
            ax.text(coord[0] * 1.1, coord[1] * 1.1, coord[2] * 1.1, label, fontsize=7, color=color)

        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=20, alpha=0.9, zorder=5)
        ax.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)')
        ax.set_zlabel(f'PC3 ({var_pct[2]:.1f}%)')
        ax.set_title(
            f'Token Embedding Vectors — 3D PCA  (top {num_top} tokens by frequency)\n'
            f'Colored by token length  |  Total variance explained: {sum(var_pct):.1f}%'
        )

        scalar_mappable = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=top_indices_length.min(), vmax=top_indices_length.max()))
        scalar_mappable.set_array([])
        plt.colorbar(scalar_mappable, ax=ax, shrink=0.45, pad=0.1, label='Token length (chars)')
        plt.tight_layout()
        plt.show()

