import pickle
import os

class CheckpointManager:
    def __init__(self, file_name, checkpoint_dir="checkpoints"):
        self.file_name = file_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, file_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save_checkpoint(self, model, vocabulary, epoch_summaries):
        file_path = os.path.join(self.checkpoint_dir, self.file_name)
        with open(file_path, "wb") as file:
            pickle.dump({
                "model": model,
                "vocabulary": vocabulary,
                "epoch_summaries": epoch_summaries,
            }, file)
        print(f"  Checkpoint saved -> {file_path}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as file:
                checkpoint = pickle.load(file)
            best_model = checkpoint["model"]
            epoch_summaries = checkpoint["epoch_summaries"]
            best_perplexity = min(summary["test_perplexity"] for summary in epoch_summaries)
            
            print(f"Loaded checkpoint, best perplexity: {best_perplexity:.4f}")
            return best_model.clone(), epoch_summaries, best_perplexity
        else:
            print("No checkpoint found, starting from scratch.")
            return None, [], float("inf")