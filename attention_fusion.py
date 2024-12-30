import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class SelfAttention(tf.keras.layers.Layer):
    """
    Self-attention layer for feature weighting.
    """
    def __init__(self, attention_units=256, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_units = attention_units

    def build(self, input_shape):
        self.W1 = self.add_weight(
            name='attention_weight1',
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='attention_weight2',
            shape=(self.attention_units, 1),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Calculate attention scores
        attention = tf.tanh(tf.matmul(inputs, self.W1))
        attention_weights = tf.nn.softmax(tf.matmul(attention, self.W2), axis=1)

        # Apply attention weights to input
        attended_features = inputs * attention_weights
        return attended_features

def create_attention_model(feature_extractors, num_classes):
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

    # Extract and concatenate features
    features = []
    for extractor in feature_extractors:
        x = extractor(input_layer)
        features.append(x)

    # Concatenate features
    if len(features) > 1:
        concatenated = tf.keras.layers.Concatenate()(features)
    else:
        concatenated = features[0]

    # Reshape for attention layer
    x = tf.keras.layers.Reshape((1, concatenated.shape[-1]))(concatenated)

    # Apply attention
    attended = SelfAttention()(x)

    # Flatten and add classification layers
    x = tf.keras.layers.Flatten()(attended)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=outputs, name="attention_fused_model")
    return model