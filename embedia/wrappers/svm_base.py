"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

SVM wrapper base classes — algorithm contracts independent of ML library.

Hierarchy:
    ClassifierWrapperBase
        └── SVMWrapperBase
                ├── SVMKernelWrapperBase   (SVC — explicit support vectors)
                └── SVMDirectWrapperBase   (LinearSVC — direct weight matrix)

For sklearn implementations see: wrappers/sklearn/svm.py
"""

import numpy as np
from embedia.core.layer_wrapper import ClassifierWrapperBase, OutputPredictionType


# ─────────────────────────────────────────────────────────────────────────────
# Strategy constants
# ─────────────────────────────────────────────────────────────────────────────

class SVMStrategy:
    """
    Strategy identifiers for multiclass SVM inference.
    Used by SvmBaseLayer.required_files to select the correct C implementation.
    """
    OVO = 'ovo'   # One-vs-One  — sparse coefficients, majority voting
    OVR = 'ovr'   # One-vs-Rest — dense  coefficients, argmax of scores


# ─────────────────────────────────────────────────────────────────────────────
# SVM contract
# ─────────────────────────────────────────────────────────────────────────────

class SVMWrapperBase(ClassifierWrapperBase):
    """
    Base contract for all SVM classifiers.

    Adds kernel configuration and intercepts, which are common to both
    the kernel variant (explicit support vectors) and the direct variant
    (learned weight matrix).
    """

    @property
    def output_prediction_type(self) -> OutputPredictionType:
        """SVM outputs raw scores — argmax gives the predicted class."""
        return OutputPredictionType.CLASS_PROBABILITIES

    @property
    def kernel_type(self) -> str:
        """
        Kernel variant: 'linear', 'poly', 'rbf', or 'sigmoid'.
        Used by the code generator to select the correct C function.
        """
        raise NotImplementedError

    @property
    def kernel_params(self) -> tuple:
        """
        Kernel hyperparameters as (gamma, intercept, degree).
        - gamma:     scale factor used by poly, RBF and sigmoid kernels.
        - intercept: bias term used by poly and sigmoid kernels.
        - degree:    exponent used by the polynomial kernel.
        Unused parameters are still returned (set to 0 / 1 as appropriate).
        """
        raise NotImplementedError

    @property
    def intercepts(self) -> np.ndarray:
        """
        Intercept (bias) for each binary classifier.
        Shape depends on strategy:
            OvO: (n_pairs,)    — one per class pair
            OvR: (n_classes,)  — one per class
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Kernel SVM contract  (SVC — explicit support vectors)
# ─────────────────────────────────────────────────────────────────────────────

class SVMKernelWrapperBase(SVMWrapperBase):
    """
    Contract for kernel SVM classifiers that store explicit support vectors.

    Covers both OvO and OvR strategies. The active strategy is indicated by
    the `strategy` property so that SvmBaseLayer can select the correct C
    files (svm_ovo.h/c or svm_ovr.h/c) and the code generator can interpret
    `coefficients` correctly.

    Coefficient layout:
        OvO: (n_pairs,   n_SV) — coef[pair, sv] != 0 only if sv participates
        OvR: (n_classes, n_SV) — coef[class, sv] for all support vectors
    """

    @property
    def strategy(self) -> str:
        """
        Multiclass strategy: SVMStrategy.OVO or SVMStrategy.OVR.
        Determines C file selection and coefficient interpretation.
        """
        raise NotImplementedError

    @property
    def n_support_vectors(self) -> int:
        """Total number of support vectors across all classes."""
        raise NotImplementedError

    @property
    def support_vectors(self) -> np.ndarray:
        """
        Support vector matrix.
        Shape: (n_SV, n_features), dtype float32.
        """
        raise NotImplementedError

    @property
    def coefficients(self) -> np.ndarray:
        """
        Dual coefficients matrix.
        Shape and semantics depend on strategy (see class docstring).
        dtype: float32.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Direct linear SVM contract  (LinearSVC — no support vectors)
# ─────────────────────────────────────────────────────────────────────────────

class SVMDirectWrapperBase(SVMWrapperBase):
    """
    Contract for linear SVM classifiers that store weight vectors directly.

    Equivalent to sklearn's LinearSVC. No support vectors, no kernel
    evaluation — inference is a plain dot product per class.

    Always uses OvR strategy. C implementation: svm_ovr.h / svm_ovr.c,
    function svm_direct_classifier_layer().

    Coefficient layout:
        coefficients: (n_classes, n_features) — coef[class, feature]
        intercepts:   (n_classes,)
    """

    @property
    def strategy(self) -> str:
        """Always OvR — LinearSVC does not support OvO."""
        return SVMStrategy.OVR

    @property
    def kernel_type(self) -> str:
        """Always linear — no kernel trick."""
        return 'linear'

    @property
    def kernel_params(self) -> tuple:
        """Kernel params unused — returns neutral values (0, 0, 1)."""
        return (0.0, 0.0, 1)

    @property
    def coefficients(self) -> np.ndarray:
        """
        Weight matrix learned directly during training.
        Shape: (n_classes, n_features), dtype float32.
        coef[i, f] = weight of feature f in the binary classifier for class i.
        """
        raise NotImplementedError