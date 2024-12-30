
from pathlib import Path
import pandas as pd
import tensorflow as tf 


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.checkpoint_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)

        self.model_dir = self.checkpoint_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        self.results_dir = self.checkpoint_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)

    def save_model(self, model, model_name:str):
        model.save(self.model_dir / f"{model_name}.keras")

    def load_model(self, model_name:str, custom: dict=None):
        model_path = self.model_dir / f"{model_name}.keras"
        if model_path.exists():
            return tf.keras.models.load_model(model_path, custom_objects=custom)
        return None

    def save_results(self, results, model_name:str):
        pd.DataFrame(results).to_csv(self.results_dir / f"{model_name}_results.csv", index=False)

    def load_results(self, model_name:str):
        return pd.read_csv(self.results_dir / f"{model_name}_results.csv")
