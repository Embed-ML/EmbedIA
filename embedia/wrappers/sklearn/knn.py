"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

sklearn distance-based classifier wrapper implementation.
"""

import numpy as np
from embedia.core.layer_wrapper import OutputPredictionType
from embedia.wrappers.distance_base import DistanceWrapperBase


class SKLKnnWrapper(DistanceWrapperBase):
    """Wraps sklearn's KNeighborsClassifier."""

    SUPPORTED_DISTANCES = [
        'euclidean', 'manhattan', 'cosine',
        'chebyshev', 'braycurtis', 'canberra'
    ]

    @property
    def n_classes(self) -> int:
        return len(self._target.classes_)

    @property
    def n_features(self) -> int:
        return self._target.n_features_in_

    @property
    def output_prediction_type(self) -> OutputPredictionType:
        return OutputPredictionType.CLASS_PROBABILITIES

    @property
    def n_samples(self) -> int:
        return self._target.n_samples_fit_

    @property
    def n_neighbors(self) -> int:
        return self._target.n_neighbors

    @property
    def fit_data(self) -> np.ndarray:
        return self._target._fit_X.astype(np.float32)

    @property
    def fit_target(self) -> np.ndarray:
        return self._target._y.astype(np.uint16)

    @property
    def distance_function(self) -> str:
        metric     = self._target.metric
        p          = self._target.p
        normalized = self._normalize_distance(metric, p)

        if normalized not in self.SUPPORTED_DISTANCES:
            raise ValueError(
                f"Unsupported distance '{normalized}' for KNN. "
                f"Supported: {self.SUPPORTED_DISTANCES}"
            )
        if normalized == 'euclidean':
            normalized = normalized+'_sq'
        return normalized

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _normalize_distance(self, metric: str, p: int) -> str:
        """Map sklearn metric aliases to EmbedIA canonical names."""
        metric = metric.lower()
        if metric in ('manhattan', 'cityblock', 'l1') or (metric == 'minkowski' and p == 1):
            return 'manhattan'
        if metric in ('euclidean', 'l2') or (metric == 'minkowski' and p == 2):
            return 'euclidean'
        return metric