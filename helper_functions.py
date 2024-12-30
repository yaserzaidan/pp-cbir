import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(classes, y_true, preds, save_path: str=None, title: str=''):
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )
    disp.plot()
    if save_path:
        save_path.mkdir(exist_ok=True)
        plt.savefig(save_path / f'confusion_matrix_{title}.pdf')
        plt.savefig(save_path / f'confusion_matrix_{title}.svg')
    plt.show()
    

def extract_features(model, layer_name='global_average_pooling2d'):
    """
    Extract features from a specific layer of the model.

    Args:
        model: Pre-trained model
        layer_name: Name of the layer to extract features from

    Returns:
        Model that outputs features from the specified layer
    """
    # Get the specified layer's output
    feature_layer = model.get_layer(layer_name)

    # Create a new model that outputs features
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=feature_layer.output,
        name=f"{model.name}_feature_extractor"
    )

    # Freeze the feature extractor
    feature_extractor.trainable = False

    return feature_extractor

def plot_images(images, cmap=None):
    """
    Plots a list of images in a grid with no more than three columns.

    Parameters:
    - images (list): List of images as arrays.
    - cmap (str, optional): Colormap for displaying the images (e.g., 'gray').
    """
    num_images = len(images)
    columns = 3
    rows = (num_images + columns - 1) // columns  # Calculate the required rows

    # Set up the figure
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each image
    for i in range(len(images)):
        axes[i].imshow(images[i].astype('uint8'), cmap=cmap)
        axes[i].axis('off')  # Turn off the axes for cleaner visualization

    # Turn off any remaining axes
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
