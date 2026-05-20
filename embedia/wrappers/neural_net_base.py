"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

Neural Network wrapper base classes — algorithm contracts independent of ML library.

Defines abstract interfaces for neural network layer types:
- Dense layers
- Convolutional layers (1D, 2D, Separable)
- Pooling layers
- Activation functions
- Batch normalization
- Padding layers

For TensorFlow implementations see: wrappers/tensorflow/

Hierarchy:
    LayerWrapper (from core.layer_wrapper)
        ├── NeuralNetWrapperBase (base для all NN layers)
        │   ├── DenseWrapperBase
        │   ├── ConvolutionalWrapperBase
        │   │   ├── Conv1DWrapperBase
        │   │   ├── Conv2DWrapperBase
        │   │   └── SeparableConv2DWrapperBase
        │   ├── PoolingWrapperBase
        │   ├── ActivationWrapperBase
        │   ├── BatchNormWrapperBase
        │   └── PaddingWrapperBase
"""

import numpy as np
from abc import abstractmethod
from embedia.core.layer_wrapper import LayerWrapper
from embedia.core.padding_types import PaddingType


class NeuralNetWrapperBase(LayerWrapper):
    """
    Base contract for all neural network layer wrappers.

    Common properties shared across all neural network layer types.
    """

    @property
    def input_shape(self):
        """Input shape of the layer. Shape: tuple or None."""
        raise NotImplementedError

    @property
    def output_shape(self):
        """Output shape of the layer. Shape: tuple or None."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the layer."""
        raise NotImplementedError

    @property
    def data_format(self) -> str:
        """
        Data format: 'channels_last' or 'channels_first'.
        Default is 'channels_last' for most frameworks.
        """
        return 'channels_last'

    @property
    def input_channels(self):
        """Number of input channels."""
        return None


class DenseWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for Dense (fully connected) layer wrappers.

    Dense layers perform: output = activation(input @ weights + biases)
    """

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Weight matrix for the dense layer.
        Shape: (input_features, output_features), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def biases(self) -> np.ndarray:
        """
        Bias vector for the dense layer.
        Shape: (output_features,), dtype float32.
        """
        raise NotImplementedError

    @property
    def use_bias(self) -> bool:
        """Whether the layer uses bias."""
        return True


class ConvolutionalWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for convolutional layer wrappers.

    Common interface for Conv1D, Conv2D, and Separable variants.
    """

    @property
    @abstractmethod
    def kernel_size(self):
        """Kernel size. Tuple (k,) for 1D, (k, k) for 2D, etc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def strides(self):
        """Stride values. Tuple or int."""
        raise NotImplementedError

    @property
    @abstractmethod
    def padding(self) -> PaddingType:
        """Padding type: PaddingType.VALID, PaddingType.SAME, etc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def filters(self) -> int:
        """Number of filters (output channels)."""
        raise NotImplementedError

    @property
    def use_bias(self) -> bool:
        """Whether the layer uses bias."""
        return True

    @property
    def dilation_rate(self):
        """Dilation rate for the kernel. Default: (1,...)."""
        return None

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality: 1 for Conv1D, 2 for Conv2D, 3 for Conv3D."""
        raise NotImplementedError


class Conv1DWrapperBase(ConvolutionalWrapperBase):
    """
    Base contract for 1D convolutional layer wrappers.

    Adds 1D-specific properties.
    """

    @property
    def dimensions(self) -> int:
        return 1

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        1D convolution weights.
        Expected format: (filters, input_channels, kernel_size)
        dtype: float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def biases(self) -> np.ndarray:
        """
        Bias vector. Shape: (filters,), dtype float32.
        """
        raise NotImplementedError


class Conv2DWrapperBase(ConvolutionalWrapperBase):
    """
    Base contract for 2D convolutional layer wrappers.

    Adds 2D-specific properties.
    """

    @property
    def dimensions(self) -> int:
        return 2

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        2D convolution weights.
        Expected format: (filters, input_channels, kernel_height, kernel_width)
        dtype: float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def biases(self) -> np.ndarray:
        """
        Bias vector. Shape: (filters,), dtype float32.
        """
        raise NotImplementedError


class SeparableConv2DWrapperBase(Conv2DWrapperBase):
    """
    Base contract for Separable 2D convolutional layer wrappers.

    Separable convolution = depthwise convolution + pointwise convolution.
    """

    @property
    @abstractmethod
    def depth_weights(self) -> np.ndarray:
        """
        Depthwise kernel weights.
        Expected format: (depth_multiplier, input_channels, kernel_height, kernel_width)
        dtype: float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def depth_biases(self) -> np.ndarray:
        """
        Depthwise bias vector.
        Shape: (input_channels * depth_multiplier,), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def point_weights(self) -> np.ndarray:
        """
        Pointwise (1x1) kernel weights.
        Expected format: (filters, input_channels * depth_multiplier, 1, 1)
        dtype: float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def biases(self) -> np.ndarray:
        """
        Pointwise bias vector.
        Shape: (filters,), dtype float32.
        """
        raise NotImplementedError


class PoolingWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for pooling layer wrappers.

    Supports MaxPooling, AveragePooling, and Global variants.
    """

    @property
    @abstractmethod
    def pool_size(self):
        """Pool window size. Tuple (p,) for 1D, (p, p) for 2D, etc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def strides(self):
        """Stride values. Tuple or int."""
        raise NotImplementedError

    @property
    @abstractmethod
    def padding(self) -> PaddingType:
        """Padding type: PaddingType.VALID or PaddingType.SAME."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality: 1 for Pooling1D, 2 for Pooling2D, etc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def function_name(self) -> str:
        """
        Pooling function name for code generation.
        Examples: 'max', 'average', 'global_average', 'global_max'.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_global(self) -> bool:
        """Whether this is a GlobalPooling layer."""
        raise NotImplementedError


class ActivationWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for activation function layer wrappers.

    Wraps activation layers like ReLU, LeakyReLU, Softmax, etc.
    """

    @property
    @abstractmethod
    def function_name(self) -> str:
        """
        Activation function name (lowercase, no spaces/underscores).
        Examples: 'relu', 'leakyrelu', 'sigmoid', 'tanh', 'softmax'.
        """
        raise NotImplementedError


class BatchNormWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for Batch Normalization layer wrappers.

    Batch norm applies: output = gamma * (input - moving_mean) / sqrt(moving_variance + epsilon) + beta
    """

    @property
    @abstractmethod
    def gamma(self) -> np.ndarray:
        """
        Scale (learnable) parameter.
        Shape: (channels,), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def beta(self) -> np.ndarray:
        """
        Shift (learnable) parameter.
        Shape: (channels,), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def moving_mean(self) -> np.ndarray:
        """
        Running mean (non-trainable).
        Shape: (channels,), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def moving_variance(self) -> np.ndarray:
        """
        Running variance (non-trainable).
        Shape: (channels,), dtype float32.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """
        Small constant for numerical stability (typically 1e-3 or 1e-5).
        """
        raise NotImplementedError


class PaddingWrapperBase(NeuralNetWrapperBase):
    """
    Base contract for padding layer wrappers.

    Examples: ZeroPadding1D, ZeroPadding2D, etc.
    """

    @property
    @abstractmethod
    def padding(self):
        """
        Padding specification.
        Can be tuple, int, or PaddingType depending on implementation.
        """
        raise NotImplementedError

