"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

sklearn decision tree wrapper implementation.
"""

import numpy as np
from embedia.core.layer_wrapper import OutputPredictionType
from embedia.wrappers.tree_base import TreeWrapperBase


class SKLDecisionTreeClassifierWrapper(TreeWrapperBase):
    """Wraps sklearn's DecisionTreeClassifier."""

    @property
    def n_classes(self) -> int:
        return self._target.n_classes_

    @property
    def n_features(self) -> int:
        return self._target.n_features_in_

    @property
    def output_prediction_type(self) -> OutputPredictionType:
        return OutputPredictionType.DIRECT_CLASS_ID

    @property
    def node_count(self) -> int:
        return self._target.tree_.node_count

    @property
    def node_features(self) -> np.ndarray:
        """Leaf nodes remapped from -2 to -1 for consistency."""
        features = self._target.tree_.feature.copy()
        features[features < 0] = -1
        return features

    @property
    def node_thresholds(self) -> np.ndarray:
        return self._target.tree_.threshold

    @property
    def node_values(self) -> np.ndarray:
        return np.array([v[0].argmax() for v in self._target.tree_.value])

    @property
    def node_left(self) -> np.ndarray:
        return self._target.tree_.children_left

    @property
    def node_right(self) -> np.ndarray:
        return self._target.tree_.children_right

    @property
    def input_shape(self) -> tuple:
        return (None, self._target.tree_.n_features)

    @property
    def output_shape(self) -> tuple:
        return (None, 1)