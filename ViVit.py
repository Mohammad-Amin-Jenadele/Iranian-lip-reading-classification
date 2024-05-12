from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TubeletEmbedding(layers.Layer):
    """
    A custom Keras layer for creating embeddings of video tubelets.

    This layer projects input video tubelets into embeddings using 3D convolutional operations
    followed by flattening.

    Args:
        embed_dim (int): The dimensionality of the output embeddings.
        patch_size (int or tuple): The size of the patches to be used for convolutional projection.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.

    Attributes:
        projection (tf.keras.layers.Conv3D): A 3D convolutional layer for projecting input tubelets.
        flatten (tf.keras.layers.Reshape): A reshape layer to flatten the projected patches.

    Methods:
        call(self, videos): Forward pass method to compute embeddings of input videos.

    Inputs:
        videos (tf.Tensor): A tensor representing input video tubelets. The expected shape 
            of this tensor would typically be (batch_size,num_frames, height, width, channels),
            where:
                - batch_size: The number of samples in the batch.
                - height: The height of each video frame.
                - width: The width of each video frame.
                - num_frames:Duration of the video (number of frames).
                - channels: The number of channels in each frame (e.g., RGB channels).

    Outputs:
        flattened_patches (tf.Tensor): A tensor representing the flattened embeddings of 
            the input videos. The shape of this tensor would be (batch_size, num_patches, 
            embed_dim), where:
                - batch_size: The same as the input, representing the number of samples 
                  in the batch.
                - num_patches: The number of patches after convolution and flattening.
                - embed_dim: The dimensionality of the output embeddings, specified 
                  during the initialization of the TubeletEmbedding layer.
    """

    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        """
        Compute embeddings of input videos.

        Args:
            videos (tf.Tensor): Input video tubelets.

        Returns:
            flattened_patches (tf.Tensor): Flattened embeddings of the input videos.
        """
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
    



class PositionalEncoder(layers.Layer):
    """
    A custom Keras layer for adding positional encoding to token embeddings.

    This layer adds positional encoding to input token embeddings by adding learned
    positional embeddings to the token embeddings.

    Args:
        embed_dim (int): The dimensionality of the token embeddings.
        **kwargs: Additional keyword arguments to be passed to the base class constructor.

    Attributes:
        embed_dim (int): The dimensionality of the token embeddings.
        position_embedding (tf.keras.layers.Embedding): A trainable embedding layer for
            representing positional embeddings.
        positions (tf.Tensor): A tensor representing positions of tokens.

    Methods:
        build(self, input_shape): Method to build the layer and create the necessary
            sublayers and tensors.
        call(self, encoded_tokens): Method for performing the forward pass of the layer
            and computing positional encodings for the input token embeddings.

    Inputs:
        encoded_tokens (tf.Tensor): A tensor representing input token embeddings.
            The shape of this tensor should be (batch_size, num_tokens, embed_dim), where:
                - batch_size: The number of samples in the batch.
                - num_tokens: The number of tokens in each sample.
                - embed_dim: The dimensionality of the token embeddings.

    Outputs:
        encoded_tokens (tf.Tensor): A tensor representing token embeddings with positional
            encoding added. The shape of this tensor is the same as the input tensor,
            (batch_size, num_tokens, embed_dim).
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        """
        Build the layer and create the necessary sublayers and tensors.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None
        """
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = np.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        """
        Perform the forward pass of the layer and compute positional encodings
        for the input token embeddings.

        Args:
            encoded_tokens (tf.Tensor): Input token embeddings.

        Returns:
            encoded_tokens (tf.Tensor): Token embeddings with positional encoding added.
        """
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens




def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape,
    transformer_layers,
    num_heads,
    embed_dim,
    layer_norm_eps,
    num_classes,
):
    """
    Function to create a ViViT (Vision Vision Transformers) classifier model.

    Args:
        tubelet_embedder (TubeletEmbedding): A tubelet embedding layer to project input videos into embeddings.
        positional_encoder (PositionalEncoder): A positional encoder layer to add positional encoding to token embeddings.
        input_shape (tuple, optional): The shape of the input tensor
        transformer_layers (int, optional): The number of transformer layers
        num_heads (int, optional): The number of attention heads.
        embed_dim (int, optional): The dimensionality of the embeddings
        layer_norm_eps (float, optional): Epsilon value for layer normalization
        num_classes (int, optional): The number of classes for classification

    Returns:
        tf.keras.Model: A Keras model representing the ViViT classifier.
    """

    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=keras.activations.gelu),
                layers.Dense(units=embed_dim, activation=keras.activations.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


