"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

TensorFlow base wrapper — common utilities for all TensorFlow layer wrappers.

This module provides TensorflowWrapper, the base class for all TensorFlow/Keras
layer wrappers, including common properties like input_shape, output_shape,
data_format detection, and padding standardization.

For specific layer implementations see:
- convolutional.py
- dense.py
- pooling.py
- activation.py
- batch_normalization.py
- padding.py
"""

from embedia.wrappers.neural_net_base import NeuralNetWrapperBase
from embedia.core.padding_types import PaddingType


class TensorflowWrapper(NeuralNetWrapperBase):
    """
    Base wrapper for all TensorFlow/Keras layers.

    Provides common interface for introspecting layer properties:
    - Shape information (input/output)
    - Data format detection (channels_last/channels_first)
    - Padding standardization
    - Activation access
    """

    @property
    def input_shape(self):
        """
        Input shape of the layer.

        Returns tuple if available, None otherwise.
        Tries both symbolic shape (.input.shape) and static shape (.input_shape).
        """
        if hasattr(self._target, "input"):
            return self._target.input.shape
        elif hasattr(self._target, "input_shape"):
            return self._target.input_shape
        else:
            return None

    @property
    def output_shape(self):
        """
        Output shape of the layer.

        Returns tuple if available, None otherwise.
        Tries both symbolic shape (.output.shape) and static shape (.output_shape).
        """
        if hasattr(self._target, "output"):
            return self._target.output.shape
        elif hasattr(self._target, "output_shape"):
            return self._target.output_shape
        else:
            return None

    @property
    def name(self):
        """Layer name."""
        return self._target.name

    @property
    def activation(self):
        """
        Activation function if available.

        Some layers have an 'activation' attribute, others don't.
        Returns None if not present.
        """
        if hasattr(self._target, 'activation'):
            return self._target.activation
        return None

    @property
    def input_channels(self):
        """
        Number of input channels.

        Location depends on data format:
        - channels_last:  last axis
        - channels_first: axis 1 (after batch)
        """
        shape = self.input_shape
        if shape is None:
            return None
        if self.data_format == 'channels_last':
            return shape[-1]
        else:  # channels_first
            return shape[1]  # after batch dimension

    @property
    def data_format(self):
        """
        Data format: 'channels_last' or 'channels_first'.

        Detection strategy:
        1. Check layer's data_format attribute
        2. Infer from input_shape for 4D tensors
        3. Default to 'channels_last' (most common in TF/Keras)
        """
        # Most layers have this attribute
        if hasattr(self._target, 'data_format'):
            return self._target.data_format

        # Fallback: infer from input_shape
        if self.input_shape and len(self.input_shape) == 4:
            # (None, h, w, c) → channels_last
            # (None, c, h, w) → channels_first
            if self.input_shape[3] is not None and isinstance(self.input_shape[3], int):
                return 'channels_last'
            elif self.input_shape[1] is not None and isinstance(self.input_shape[1], int):
                return 'channels_first'

        return 'channels_last'  # default

    def _standardize_padding(self, raw_padding=None):
        """
        Convert padding specification to standardized PaddingType.

        Handles TensorFlow's string padding ('valid', 'same', 'causal')
        and converts to EmbedIA's PaddingType enum.

        Parameters
        ----------
        raw_padding : str, tuple, int, or None
            Padding specification from TensorFlow layer.
            If None, uses the layer's 'padding' attribute (defaults to 'valid').

        Returns
        -------
        PaddingType or original value
            Standardized padding type, or original value if not a string.
        """
        if raw_padding is None:
            raw_padding = getattr(self._target, 'padding', 'valid')

        if isinstance(raw_padding, str):
            padding_map = {
                'valid': PaddingType.VALID,
                'same': PaddingType.SAME,
                'causal': PaddingType.CAUSAL
            }
            return padding_map.get(raw_padding.lower(), PaddingType.VALID)

        # Keep tuples or ints as-is
        return raw_padding

