"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

Normalizer wrapper base class — algorithm contract independent of ML library.

For sklearn implementations see: wrappers/sklearn/normalizer.py
"""

import numpy as np
from embedia.core.layer_wrapper import LayerWrapper


class NormalizerWrapperBase(LayerWrapper):
    """
    Base contract for all normalizer/scaler wrappers.

    Normalization is expressed as:
        output = (input - sub_values) / div_values

    Subclasses that only divide (e.g. MaxAbsScaler) return None for
    sub_values — the implementation treats None as zero subtraction.
    """

    @property
    def div_values(self) -> np.ndarray:
        """Values to divide by. Shape: (n_features,)."""
        raise NotImplementedError

    @property
    def sub_values(self):
        """
        Values to subtract before dividing. Shape: (n_features,).
        Returns None if no subtraction is needed (e.g. MaxAbsScaler).
        """
        return None

    @property
    def function_name(self) -> str:
        """
        Name of the C normalization function to call.
        Examples: 'min_max', 'max_abs', 'standard', 'robust'.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple:
        return self.div_values.shape

    @property
    def output_shape(self) -> tuple:
        return self.div_values.shape