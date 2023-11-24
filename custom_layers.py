"""Custom layers and modules used by trading models found in hypermodels.py."""

__author__ = 'Dante St. Prix'

import math

import tensorflow as tf
from keras.layers.attention.multi_head_attention import _build_attention_equation

@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """Applies slight modifications to the Keras implementation for improved distributed efficiency and stability."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity_layer = tf.keras.layers.Lambda(lambda x: x)  # For converting back to float16.

    def _build_attention(self, rank):
        """As opposed to the original function, this function builds a softmax layer with dtype='float32' for improved
        numerical stability with mixed precision training.
        Refer to original implementation for more details.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (self._dot_product_equation, self._combine_equation, attn_scores_rank,) = \
            _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = tf.keras.layers.Softmax(axis=norm_axes, dtype='float32')
        self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        """Due to a bug, when using MultiWorkerMirroredStrategy, this function creates a synchronization point and
        causes the GPUs to stagger, resulting in >10% additional idle time. The softmax function is placed in a context
        manager to prevent it from being compiled with XLA.
        Refer to original implementation for more details.
        """
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        with tf.xla.experimental.jit_scope(compile_ops=False):
            # a kernel.
            attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_scores = self.identity_layer(attention_scores)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores


@tf.keras.utils.register_keras_serializable()
class TransformerModule(tf.keras.layers.Layer):
    """An implementation of a transformer module as described in the 'Attention Is All You Need' paper.

    This module sequentially combines the following layers:
    1) MultiHeadAttention (self-attention) Layer with skip connections.
    2) Layer normalization.
    3) 2 feedforward layers with skip connections.
    4) Dropout layer.
    5) Layer normalization.

    Args:
      num_heads: Number of attention heads (int).
      key_dim: Size of each attention head for query and key (int).
      kernel_initializer: Initializer for feedforward layers.
      kernel_regularizer: Regularizer for MultiHeadAttention and feedforward layers (Regularizer).
      dropout_rate: Dropout probability (float).
      activation: Activation function for 2nd feedforward layer (Activation).
    """
    def __init__(self, num_heads, key_dim, kernel_initializer, kernel_regularizer, dropout_rate, activation, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.initializer = kernel_initializer
        self.regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.add_layer = tf.keras.layers.Add()
        self.layers_ = []

    def build(self, input_shape):
        self.layers_ = []
        self.layers_.append(MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim,
                                               dropout=self.dropout_rate, kernel_regularizer=self.regularizer))
        self.layers_.append(tf.keras.layers.LayerNormalization())
        self.layers_.append(tf.keras.layers.Dense(512, activation=self.activation, kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer))
        self.layers_.append(tf.keras.layers.Dense(self.key_dim, kernel_initializer=self.initializer))
        self.layers_.append(tf.keras.layers.Dropout(self.dropout_rate))
        self.layers_.append(tf.keras.layers.LayerNormalization())

    def call(self, inputs, training=None):
        z = inputs
        skip = z
        z = self.layers_[0](z, z, training=training)  # Multi-head attention (self-attention).
        z = self.layers_[1](z)  # Layer normalization 1.
        z = self.add_layer([z, skip])
        skip = z
        z = self.layers_[2](z)  # Feedforward 1.
        z = self.layers_[3](z)  # Feedforward 2.
        z = self.layers_[4](z, training=training)  # Dropout.
        z = self.add_layer([z, skip])
        z = self.layers_[5](z)  # Layer normalization 2.
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'initializer': self.initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config

@tf.keras.utils.register_keras_serializable()
class GatedActivationUnit(tf.keras.layers.Layer):
    """Regulates the information flow by using tanh (value) and sigmoid (scaling) activations.

    Found by the authors of the original WaveNet paper to outperform ReLU."""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get('tanh')

    def call(self, inputs, training=False):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = tf.keras.activations.sigmoid(inputs[..., n_filters:])
        return linear_output * gate

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class WavenetResidualBlock(tf.keras.layers.Layer):
    """The residual block used in WaveNet-based models.
    Includes
    1) Layer normalization.
    2) Dilated convolution: Captures long-range dependencies in the input data. The dilation rate determine the spacing
     between the kernels.
    3) Gated activation unit: Regulates information flow.
    4) Dropout (optional).
    5) 1x1 convolution: Ensures the number of output channels is the same as 'filters'.

    Args:
        filters: A int value representing the number of filters for the convolutional layers (int).
        padding: Type of padding for the convolutional layers (should be 'causal' or 'same'), (str).
        dilation_rate: Spacing between the kernel points in the dilated convolutional layer (int).
        kernel_regularizer: Regularization applied to the convolutional layers (Regularizer).
        dropout: A boolean specifying whether to include the dropout layer (bool).
        dropout_rate: Dropout probability (float).
    """
    def __init__(self, filters, padding, dilation_rate, kernel_regularizer, dropout, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.regularizer = kernel_regularizer
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.layers_ = []

    def build(self, input_shape):
        self.layers_ = []
        self.layers_.append(tf.keras.layers.LayerNormalization())
        self.layers_.append(tf.keras.layers.Conv1D(2*self.filters, kernel_size=2, padding=self.padding,
                                                   dilation_rate=self.dilation_rate,
                                                   kernel_regularizer=self.regularizer))
        self.layers_.append(GatedActivationUnit())
        if self.dropout:
            self.layers_.append(tf.keras.layers.Dropout(self.dropout_rate))
        self.layers_.append(tf.keras.layers.Conv1D(self.filters, kernel_size=1, kernel_regularizer=self.regularizer))
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Returns:
            A tuple containing the sum of the inputs and the result of the computation, and the normalized input.
        """
        z = self.layers_[0](inputs)  # Layer normalization.
        z = self.layers_[1](z)  # Dilated convolution.
        z = self.layers_[2](z)  # Gated activation unit.
        if self.dropout:
            z = self.layers_[3](z, training=training)  # Dropout layer.
        z = self.layers_[-1](z)  # 1x1 convolution.
        return tf.keras.layers.Add()([z, inputs]), z

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'regularizer': self.regularizer,
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate
        })
        return config

class MCDropout(tf.keras.layers.Dropout):
    """A dropout layer that is always active (even when training == False).

    Replaces dropout layer in Monte Carlo versions of each model.
    """
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)