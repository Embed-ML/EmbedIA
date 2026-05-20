"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow wrapper implementations.

This package organizes TensorFlow/Keras layer wrappers by layer type:
- activation.py     — activation layers (ReLU, LeakyReLU, Sigmoid, etc.)
- batch_normalization.py — batch normalization layers
- convolution.py    — convolutional layers (Conv1D, Conv2D, SeparableConv2D)
- dense.py          — fully connected (Dense) layers
- padding.py        — padding layers (ZeroPadding1D, ZeroPadding2D, etc.)
- pooling.py        — pooling layers (MaxPooling, AveragePooling, Global variants)
- base.py           — common TensorflowWrapper base class

Each wrapper inherits from both:
1. TensorflowWrapper (common TF utilities)
2. Corresponding abstract base from neural_net_base.py

For adding new TensorFlow layer types, see neural_net_base.py for interfaces.
"""

# Base utilities
from embedia.wrappers.tensorflow.base import TensorflowWrapper

# Activation layers
from embedia.wrappers.tensorflow.activation import TFActivationWrapper

# Batch normalization
from embedia.wrappers.tensorflow.batch_normalization import TFBatchNormWrapper

# Convolutional layers
from embedia.wrappers.tensorflow.convolution import (
    TFConvolutionalWrapper,
    TFConv1DWrapper,
    TFConv2DWrapper,
    TFSeparableConv2DWrapper,
)

# Dense layers
from embedia.wrappers.tensorflow.dense import TFDenseWrapper

# Padding layers
from embedia.wrappers.tensorflow.padding import TFPaddingWrapper

# Pooling layers
from embedia.wrappers.tensorflow.pooling import (
    TFPoolingWrapper,
    TFPoolWrapper,  # Backward compatibility alias
)

__all__ = [
    'TensorflowWrapper',
    'TFActivationWrapper',
    'TFBatchNormWrapper',
    'TFConvolutionalWrapper',
    'TFConv1DWrapper',
    'TFConv2DWrapper',
    'TFSeparableConv2DWrapper',
    'TFDenseWrapper',
    'TFPaddingWrapper',
    'TFPoolingWrapper',
    'TFPoolWrapper',  # Backward compatibility
]

