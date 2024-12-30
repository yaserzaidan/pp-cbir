from enum import Enum
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
from tqdm.notebook import tqdm

class QuantizationType(Enum):
    MAX = 'max'
    KL = 'kl'
    PERCENTILE_99 = 'percentile_99'
    FULL = 'full'

class ModelQuantizer:
    def __init__(self, model, data_generator, quantization_type: QuantizationType):
        self.model = model
        self.data_generator = data_generator
        self.quantization_type = quantization_type
        self.temp_dir = Path(tempfile.mkdtemp())

    def _representative_dataset(self):
        """Generate representative dataset for quantization calibration"""
        for i in range(min(10, len(self.data_generator))):
            batch_x, _ = self.data_generator[i]
            for image in batch_x:
                # Convert to float32
                image = tf.cast(image, tf.float32)
                yield [np.expand_dims(image, axis=0).astype(np.float32)]

    def _save_keras_model(self):
        model_path = self.temp_dir / "temp_model"
        self.model.save(model_path)
        return model_path

    def _get_base_converter(self):
        """Create base TFLite converter with common settings"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_dataset
        # Ensure float32 input
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        return converter

    def _apply_max_quantization(self):
        converter = self._get_base_converter()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return converter.convert()

    def _apply_kl_quantization(self):
        converter = self._get_base_converter()
        converter._experimental_new_quantizer = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return converter.convert()

    def _apply_percentile_quantization(self):
        converter = self._get_base_converter()
        converter._experimental_new_quantizer = True
        converter._experimental_calibration_percentile = 99.0
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return converter.convert()

    def _apply_full_quantization(self):
        converter = self._get_base_converter()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return converter.convert()

    def quantize(self):
        quantization_methods = {
            QuantizationType.MAX: self._apply_max_quantization,
            QuantizationType.KL: self._apply_kl_quantization,
            QuantizationType.PERCENTILE_99: self._apply_percentile_quantization,
            QuantizationType.FULL: self._apply_full_quantization
        }

        quantize_fn = quantization_methods.get(self.quantization_type)
        if not quantize_fn:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")

        return quantize_fn()


class QuantizedModelManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.quantized_dir = self.checkpoint_dir / 'quantized_models'
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

    def save_quantized_model(self, quantized_model, model_name, quantization_type):
        output_path = self.quantized_dir / f"{model_name}_{quantization_type.value}.tflite"
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        return output_path

    def load_quantized_model(self, model_name, quantization_type):
        model_path = self.quantized_dir / f"{model_name}_{quantization_type.value}.tflite"
        if not model_path.exists():
            return None
        return tf.lite.Interpreter(model_path=str(model_path))

def quantize_models(base_models, data_module, checkpoint_dir):
    quantized_manager = QuantizedModelManager(checkpoint_dir)
    results = {}

    for model in base_models:
        model_results = {}
        for quant_type in QuantizationType:
            print(f"Quantizing {model.name} using {quant_type.value} quantization...")

            quantizer = ModelQuantizer(
                model=model,
                data_generator=data_module.valid_generator,
                quantization_type=quant_type
            )

            try:
                quantized_model = quantizer.quantize()
                saved_path = quantized_manager.save_quantized_model(
                    quantized_model,
                    model.name,
                    quant_type
                )
                model_results[quant_type.value] = str(saved_path)
            except Exception as e:
                print(f"Error quantizing {model.name} with {quant_type.value}: {str(e)}")
                model_results[quant_type.value] = None

        results[model.name] = model_results

    return results

class QuantizedModelManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.quantized_dir = self.checkpoint_dir / 'quantized_models'
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

    def save_quantized_model(self, quantized_model, model_name, quantization_type):
        output_path = self.quantized_dir / f"{model_name}_{quantization_type.value}.tflite"
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        return output_path

    def load_quantized_model(self, model_name, quantization_type):
        """Load a quantized model and return an initialized interpreter"""
        model_path = self.quantized_dir / f"{model_name}_{quantization_type.value}.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"Quantized model not found at {model_path}")

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter

    def get_model_size_comparison(self, original_models, save_csv=True):
        """Compare the size of original and quantized models"""
        comparison_data = []

        # Get sizes of original models
        for model in original_models:
            # Save original model temporarily to get its size
            temp_path = self.checkpoint_dir / f"temp_{model.name}.keras"
            model.save(temp_path)
            original_size = os.path.getsize(temp_path) / (1024 * 1024)  # Convert to MB
            os.remove(temp_path)  # Clean up

            # Get sizes of quantized versions
            for quant_type in QuantizationType:
                try:
                    quantized_path = self.quantized_dir / f"{model.name}_{quant_type.value}.tflite"
                    if quantized_path.exists():
                        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # Convert to MB
                        size_reduction = ((original_size - quantized_size) / original_size) * 100

                        comparison_data.append({
                            'Model': model.name,
                            'Quantization': quant_type.value,
                            'Original Size (MB)': round(original_size, 2),
                            'Quantized Size (MB)': round(quantized_size, 2),
                            'Size Reduction (%)': round(size_reduction, 2)
                        })
                except FileNotFoundError:
                    continue

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Save to CSV if requested
        if save_csv:
            csv_path = self.checkpoint_dir / 'model_size_comparison.csv'
            df.to_csv(csv_path, index=False)

        return df

class QuantizedModelPredictor:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_input(self, image):
        """Preprocess input image to match model requirements"""
        if isinstance(image, tf.Tensor):
            image = image.numpy()

        # Ensure image is float32
        image = image.astype(np.float32)

        # Resize if needed
        required_shape = self.input_details[0]['shape'][1:3]
        if not np.array_equal(image.shape[:2], required_shape):
            image = tf.image.resize(image, required_shape)

        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        return image

    def predict_image(self, image):
        """Make prediction using the quantized model"""
        # Preprocess input
        processed_image = self.preprocess_input(image)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)

        # Run inference
        self.interpreter.invoke()

        # Get prediction
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        return prediction
    
    def predict(self, sequence):
        all_predictions = []

        for i in tqdm(range(len(sequence))):
            X_batch, _ = sequence[i]
            batch_predictions = [self.predict_image(X) for X in X_batch]
            all_predictions.extend(batch_predictions)

        return all_predictions


def evaluate_quantized_model(quantized_predictor, data_generator, num_batches=None):
    """Evaluate the quantized model on a data generator"""
    correct = 0
    total = 0

    if num_batches is None:
        num_batches = len(data_generator)

    for i in range(num_batches):
        x, y_true = data_generator[i]

        # Get predictions for batch
        predictions = []
        for image in x:
            pred = quantized_predictor.predict_image(image)
            predictions.append(pred)

        # Convert predictions to numpy array
        predictions = np.vstack(predictions)

        # Calculate accuracy
        correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_true, axis=1))
        total += len(y_true)

    return correct / total

# Example usage:
def compare_and_evaluate_models(base_models, data_module, checkpoint_dir):
    """Compare and evaluate original and quantized models"""
    quantized_manager = QuantizedModelManager(checkpoint_dir)

    # # Get size comparison
    # size_comparison = quantized_manager.get_model_size_comparison(base_models)
    # print("\nModel Size Comparison:")
    # print(size_comparison)

    # Evaluate models
    results = []
    for model in base_models:
        # Evaluate original model
        original_accuracy = model.evaluate(data_module.valid_generator, verbose=0)[1]

        # Evaluate quantized versions
        for quant_type in QuantizationType:
            try:
                # Load quantized model
                interpreter = quantized_manager.load_quantized_model(model.name, quant_type)
                predictor = QuantizedModelPredictor(interpreter)

                # Evaluate quantized model
                quantized_accuracy = evaluate_quantized_model(
                    predictor,
                    data_module.valid_generator,
                    num_batches=5  # Limit batches for faster evaluation
                )

                results.append({
                    'Model': model.name,
                    'Quantization': quant_type.value,
                    'Original Accuracy': round(original_accuracy * 100, 2),
                    'Quantized Accuracy': round(quantized_accuracy * 100, 2),
                    'Accuracy Drop': round((original_accuracy - quantized_accuracy) * 100, 2)
                })

            except FileNotFoundError:
                continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(checkpoint_dir) / 'quantization_accuracy_comparison.csv', index=False)
    return results_df