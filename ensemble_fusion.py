from scipy import stats
import numpy as np
import tensorflow as tf

class EnsembleFusion:
    """Class implementing various fusion techniques for ensemble learning."""
    def __init__(self, model_paths=None, models=None):
        if model_paths is not None:
            self.models = [tf.keras.models.load_model(path) for path in model_paths]
        elif models is not None:
            self.models = models
        else:
            raise ValueError("Either model_paths or models must be provided")

    def predict_sequence(self, sequence):
        """
        Get predictions from all models using a sequence.

        Parameters:
        -----------
        sequence : tf.keras.utils.Sequence
            Data sequence

        Returns:
        --------
        numpy.ndarray
            Predictions from all models
        """
        all_predictions = []

        for i in range(len(sequence)):
            X_batch, _ = sequence[i]
            batch_predictions = np.array([model.predict(X_batch, verbose=0) for model in self.models])
            all_predictions.append(batch_predictions)

        return np.concatenate(all_predictions, axis=1)

    def majority_voting_sequence(self, sequence):
        """
        Implement majority voting fusion using a sequence.

        Returns:
        --------
        tuple
            Final predictions and individual model predictions
        """
        predictions = self.predict_sequence(sequence)

        # Convert probabilities to class labels
        class_predictions = np.argmax(predictions, axis=2)

        # Take mode along model axis
        ensemble_predictions = stats.mode(class_predictions, axis=0)[0].squeeze()

        return ensemble_predictions, predictions

    def weighted_average_sequence(self, sequence, weights=None):
        """
        Implement weighted average fusion using a sequence.

        Parameters:
        -----------
        weights : numpy.ndarray, optional
            Weights for each model

        Returns:
        --------
        tuple
            Final predictions and individual model predictions
        """
        predictions = self.predict_sequence(sequence)

        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            weights = np.array(weights) / np.sum(weights)

        weights = weights.reshape(-1, 1, 1)
        weighted_preds = np.sum(predictions * weights, axis=0)
        ensemble_predictions = np.argmax(weighted_preds, axis=1)

        return ensemble_predictions, predictions

    def evaluate_sequence(self, sequence, weights=None):
        """
        Evaluate both fusion techniques using a sequence.

        Returns:
        --------
        dict
            Evaluation results
        """
        # Get true labels
        true_labels = []
        for i in range(len(sequence)):
            _, y_batch = sequence[i]
            true_labels.append(np.argmax(y_batch, axis=1))
        true_labels = np.concatenate(true_labels)

        # Get predictions
        majority_preds, individual_preds_maj = self.majority_voting_sequence(sequence)
        weighted_preds, individual_preds_wav = self.weighted_average_sequence(sequence, weights)

        # Calculate accuracies
        majority_acc = np.mean(majority_preds == true_labels)
        weighted_acc = np.mean(weighted_preds == true_labels)

        # Calculate individual model accuracies
        individual_preds = np.argmax(individual_preds_maj, axis=2)
        individual_acc = [np.mean(preds == true_labels) for preds in individual_preds]

        return {
            'majority_voting_accuracy': majority_acc,
            'weighted_average_accuracy': weighted_acc,
            'individual_accuracies': individual_acc
        }