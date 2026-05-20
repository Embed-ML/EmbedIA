"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

Distance-based classifier wrapper base class — algorithm contract independent
of ML library.

For sklearn implementations see: wrappers/sklearn/distance.py
"""

import numpy as np
from embedia.core.layer_wrapper import ClassifierWrapperBase


class DistanceWrapperBase(ClassifierWrapperBase):
    """
    Base contract for distance-based classifiers (e.g. KNN).

    These classifiers store the training data at inference time and
    classify new samples by computing distances to stored samples.
    """

    @property
    def n_samples(self) -> int:
        """Number of training samples stored in the model."""
        raise NotImplementedError

    @property
    def fit_data(self) -> np.ndarray:
        """
        Training feature matrix.
        Shape: (n_samples, n_features), dtype float32.
        """
        raise NotImplementedError

    @property
    def fit_target(self) -> np.ndarray:
        """
        Training labels (class indices).
        Shape: (n_samples,), dtype uint16.
        """
        raise NotImplementedError

    @property
    def distance_function(self) -> str:
        """
        Name of the distance function to use at inference time.
        Supported values: 'euclidean', 'manhattan', 'cosine',
                          'chebyshev', 'braycurtis', 'canberra'.
        """
        raise NotImplementedError