"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

sklearn normalizer wrapper implementations.
"""

import numpy as np
from embedia.wrappers.normalizer_base import NormalizerWrapperBase


class SKLMinMaxScalerWrapper(NormalizerWrapperBase):
    """Wraps sklearn's MinMaxScaler."""

    @property
    def div_values(self) -> np.ndarray:
        return self._target.data_range_

    @property
    def sub_values(self) -> np.ndarray:
        return self._target.data_min_

    @property
    def function_name(self) -> str:
        return 'min_max'


class SKLMaxAbsScalerWrapper(NormalizerWrapperBase):
    """Wraps sklearn's MaxAbsScaler. No subtraction needed."""

    @property
    def div_values(self) -> np.ndarray:
        return self._target.max_abs_

    @property
    def function_name(self) -> str:
        return 'max_abs'


class SKLStandardScalerWrapper(NormalizerWrapperBase):
    """Wraps sklearn's StandardScaler."""

    @property
    def div_values(self) -> np.ndarray:
        return self._target.scale_

    @property
    def sub_values(self) -> np.ndarray:
        return self._target.mean_

    @property
    def function_name(self) -> str:
        return 'standard'


class SKLRobustScalerWrapper(NormalizerWrapperBase):
    """Wraps sklearn's RobustScaler."""

    @property
    def div_values(self) -> np.ndarray:
        return self._target.scale_

    @property
    def sub_values(self) -> np.ndarray:
        return self._target.center_

    @property
    def function_name(self) -> str:
        return 'robust'