import cupy as cp
import math
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, training_path, validation_path):
        self.training_path = training_path
        self.validation_path = validation_path

        with open(self.training_path, "r", encoding="utf-8") as file:
            training_data = file.read()

        with open(self.validation_path, "r", encoding="utf-8") as file:
            validation_data = file.read()

        training_data = list(training_data)
        validation_data = list(validation_data)

        self.x_train = training_data[:-1]
        self.y_train = training_data[1:]

        self.x_test = validation_data[:-1]
        self.y_test = validation_data[1:]

        self.vocabulary = sorted(set(self.x_train).union(set(self.x_test)))

        print(f"Training data length: X={len(self.x_train)}, Y={len(self.y_train)}")
        print(f"Validation data length: X={len(self.x_test)}, Y={len(self.y_test)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def get_indices(self):
        x_train_indices = cp.asarray([self.char_to_index(char) for char in self.x_train], dtype=cp.int32)
        y_train_indices = cp.asarray([self.char_to_index(char) for char in self.y_train], dtype=cp.int32)
        x_test_indices = cp.asarray([self.char_to_index(char) for char in self.x_test], dtype=cp.int32)
        y_test_indices = cp.asarray([self.char_to_index(char) for char in self.y_test], dtype=cp.int32)
        return x_train_indices, y_train_indices, x_test_indices, y_test_indices

    def get_vocabulary(self):
        return self.vocabulary

    def char_to_index(self, char: str) -> int:
        return self.vocabulary.index(char)

    def index_to_char(self, index: int) -> str:
        return self.vocabulary[index]
    
    def plot_vocabulary(self, num_columns=22, fig_width=24, font_size=22, font_family="monospace"):
        rows = math.ceil(len(self.vocabulary) / num_columns)

        fig, ax = plt.subplots(figsize=(fig_width, max(2, rows * 0.8)))

        for index, token in enumerate(self.vocabulary):
            row, col = divmod(index, num_columns)
            display_token = {" ": "<space>", "\n": "\\n", "\t": "\\t"}.get(token, token)
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