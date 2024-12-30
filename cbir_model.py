from enum import Enum
from pathlib import Path
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from checkpoint_manager import CheckpointManager
from data_module import DataModule

class ModelNames(Enum):
    VGG16 = 'vgg16'
    RESNET50 = 'resnet50'
    DENSENET121 = 'densenet121'
    EFFICIENTNETB0 = 'efficientnetb0'


def create_model(model_name:str, num_classes=3):
    if model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif model_name == 'vgg16':
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif model_name == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    else:
        raise Exception(f"{model_name} model doesn't exist or isn't implemented yet")

    # Freeze early layers
    for layer in base_model.layers[:-10]:  # Only train last few layers
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name=model_name)

    return model


class CBIRModel:
    def __init__(self,model_name:str, data_module: DataModule, checkpoint_dir:str, custom_layer:dict=None):
        self.data_module = data_module
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history = None

        self.class_weights = self.data_module.calculate_class_weights()

        self.checkpoint_dir = self.checkpoint_dir / model_name
        if self.checkpoint_dir.exists():
            print('loading model from checkpoint.........')
            self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
            self.model = self.checkpoint_manager.load_model(f"{self.model_name}_m{self.data_module.m}", custom=custom_layer)
            self.history = self.checkpoint_manager.load_results(model_name)
        else:
            print('new model created.....................')
            self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
            self.model = create_model(model_name=model_name, num_classes=len(self.data_module.train_generator.classes))

    def train(self, epochs=10):
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_manager.model_dir / f"{self.model_name}_m{self.data_module.m}.keras",
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=2,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

        # Train model
        history = self.model.fit(
            self.data_module.train_generator,
            validation_data=self.data_module.test_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights
        )

        # Save results
        self.history = history.history
        self.checkpoint_manager.save_results(self.history, self.model_name)
    
    def get_classification_report(self):
        # Predict on validation set
        y_pred = []
        y_true = []

        for i in range(len(self.data_module.valid_generator)):
            x, y = self.data_module.valid_generator[i]
            pred = self.model.predict(x, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))

        return classification_report(y_true, y_pred)

    def plot_confusion_matrix(self):
        # Predict on validation set
        y_pred = []
        y_true = []

        for i in range(len(self.data_module.valid_generator)):
            x, y = self.data_module.valid_generator[i]
            pred = self.model.predict(x, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.data_module.valid_generator.classes
        )

        disp.plot()
        plt.savefig(self.checkpoint_manager.plots_dir / f'confusion_matrix_{self.model_name}.pdf')
        plt.savefig(self.checkpoint_manager.plots_dir / f'confusion_matrix_{self.model_name}.svg')
        plt.show()

    def plot_learning_curves(self, figsize=(12, 4)):

        if 'Unnamed: 0' in self.history.keys():
            self.history.drop(columns=['Unnamed: 0'], inplace=True)

        metrics = [m for m in self.history.keys() if not m.startswith('val_')]
        if "learning_rate" in metrics:
            metrics.remove("learning_rate")


        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        # Convert to list of axes if only one metric
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            # Plot training metric
            ax.plot(self.history[metric], label='Training')

            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in self.history:
                ax.plot(self.history[val_metric], label='Validation')

            # Customize plot
            ax.set_title(f'{metric.capitalize()} Over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.checkpoint_manager.plots_dir / f'learning_curves_{self.model_name}.pdf')
        plt.savefig(self.checkpoint_manager.plots_dir / f'learning_curves_{self.model_name}.svg')
        plt.show()

