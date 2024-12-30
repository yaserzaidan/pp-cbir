import os
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from tqdm.notebook import tqdm

from cbir_system import CBIRSystem
from data_module import DataModule


class CBIREvaluator:
    def __init__(self, cbir_system: CBIRSystem, data_module: DataModule, k: int = 5):
        self.cbir_system = cbir_system
        self.data_module = data_module
        self.k = k
        self.normalization = self.data_module.normalization

    def evaluate(self):
        print("Starting evaluation...")

        mAP = self.calculate_mean_average_precision()
        recall_k = self.calculate_recall_at_k()
        inference_time = self.measure_inference_time()

        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        print(f"Recall@{self.k}: {recall_k:.4f}")
        print(f"Average inference time: {inference_time:.4f} seconds")

    def calculate_mean_average_precision(self):
        aps = []
        for j in tqdm(range(len(self.data_module.test_generator)),desc="Calculating mean average precision...."):
            batch_x, batch_y = self.data_module.test_generator[j]
            for i in range(len(batch_x)):
                query_image = batch_x[i]
                true_label = np.argmax(batch_y[i])

                similar_images, predicted_class = self.cbir_system.query_similar_images(query_image, k=self.k)

                # Create y_true based on whether predicted_class matches the true label
                y_true = [1 if predicted_class == true_label else 0] * len(similar_images)

                # Skip if there are no positive labels in y_true
                if sum(y_true) == 0:
                    continue

                y_scores = np.linspace(1, 0, len(y_true))

                ap = average_precision_score(y_true, y_scores)
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def calculate_recall_at_k(self, k=5):
        recalls = []
        for j in tqdm(range(len(self.data_module.test_generator)), desc="Calculating recall@K...."):
            batch_x, batch_y = self.data_module.test_generator[j]
            for i in range(len(batch_x)):
                query_image = batch_x[i]
                true_label = np.argmax(batch_y[i])

                _, predicted_class = self.cbir_system.query_similar_images(query_image, k=k)

                if predicted_class == true_label:
                    recalls.append(1)
                else:
                    recalls.append(0)

        return np.mean(recalls) if recalls else 0.0


    def measure_inference_time(self, num_samples=100):
        total_time = 0
        samples = 0

        for j in tqdm(range(len(self.data_module.test_generator)), desc="Measuring inference time...."):
            batch_x, _ = self.data_module.test_generator[j]
            for i in range(len(batch_x)):
                if samples >= num_samples:
                    break
                query_image = batch_x[i]
                start_time = time.time()
                self.cbir_system.query_similar_images(query_image, k=self.k)
                total_time += time.time() - start_time
                samples += 1

        return total_time / num_samples

