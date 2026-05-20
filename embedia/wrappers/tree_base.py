"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

Decision tree wrapper base class — algorithm contract independent of ML library.

For sklearn implementations see: wrappers/sklearn/tree.py
"""

import numpy as np
from embedia.core.layer_wrapper import LayerWrapper, ClassifierWrapperBase


class TreeWrapperBase(ClassifierWrapperBase):
    """
    Base contract for decision tree classifiers.

    Exposes the internal node structure that EmbedIA's C tree implementation
    needs to traverse the tree at inference time.

    Node arrays are parallel: index i refers to the same node across all
    arrays. Leaf nodes have node_left[i] == node_right[i] == -1 and
    node_features[i] == -1.
    """

    @property
    def node_count(self) -> int:
        """Total number of nodes (internal + leaf)."""
        raise NotImplementedError

    @property
    def node_features(self) -> np.ndarray:
        """
        Feature index used for splitting at each node.
        Shape: (node_count,).
        Internal nodes: feature index >= 0.
        Leaf nodes: -1.
        """
        raise NotImplementedError

    @property
    def node_thresholds(self) -> np.ndarray:
        """
        Threshold value for splitting at each internal node.
        Shape: (node_count,).
        Leaf nodes: value is meaningless.
        """
        raise NotImplementedError

    @property
    def node_values(self) -> np.ndarray:
        """
        Predicted class index at each node.
        Shape: (node_count,).
        Meaningful for leaf nodes; internal nodes hold the majority class
        of samples that reached that node.
        """
        raise NotImplementedError

    @property
    def node_left(self) -> np.ndarray:
        """
        Index of left child for each node.
        Shape: (node_count,).
        Leaf nodes: -1.
        """
        raise NotImplementedError

    @property
    def node_right(self) -> np.ndarray:
        """
        Index of right child for each node.
        Shape: (node_count,).
        Leaf nodes: -1.
        """
        raise NotImplementedError