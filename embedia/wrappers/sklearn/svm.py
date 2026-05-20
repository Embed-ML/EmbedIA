"""
EmbedIA - Embedded Machine Learning and Neural Networks Framework

sklearn SVM wrapper implementations.

Maps sklearn's SVC and LinearSVC to EmbedIA's SVM wrapper contracts.

Classes:
    SKLSvmWrapper       — wraps sklearn SVC (kernel SVM, OvO or OvR)
    SKLSvmLinearWrapper — wraps sklearn LinearSVC (direct weights, OvR)
"""

import numpy as np
from embedia.wrappers.svm_base import (
    SVMKernelWrapperBase,
    SVMDirectWrapperBase,
    SVMStrategy,
)


class SKLSvmWrapper(SVMKernelWrapperBase):
    """
    Wraps sklearn's SVC for EmbedIA inference code generation.

    Supports all sklearn kernels (linear, poly, rbf, sigmoid) and both
    multiclass strategies (OvO, OvR). The active strategy is read from
    the trained model's decision_function_shape attribute.

    Coefficient extraction handles sklearn's compact dual_coef_ layout,
    which stores (n_classes-1, n_SV) regardless of strategy.
    """

    def __init__(self, target: object):
        super().__init__(target)
        self.target.decision_function_shape = 'ovo'
    # ─────────────────────────────────────────────────────────────────────
    # ClassifierWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        return len(self._target.classes_)

    @property
    def n_features(self) -> int:
        return self._target.n_features_in_

    # ─────────────────────────────────────────────────────────────────────
    # SVMWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def kernel_type(self) -> str:
        return self._target.kernel.lower()

    @property
    def kernel_params(self) -> tuple:
        """
        Returns (gamma, intercept, degree).
        gamma is read from the fitted _gamma attribute when available
        (sklearn computes the actual value from 'scale'/'auto' there).
        """
        gamma = (
            self._target._gamma
            if hasattr(self._target, '_gamma')
            else self._target.gamma
        )
        return (gamma, self._target.coef0, self._target.degree)

    @property
    def intercepts(self) -> np.ndarray:
        """
        Intercepts for each binary classifier.
        OvO: shape (n_pairs,)   — canonical order (0,1),(0,2),...
        OvR: shape (n_classes,) — one per class
        """
        return self._target.intercept_.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # SVMKernelWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def strategy(self) -> str:
        """
        Reads decision_function_shape from the trained model.
        Returns SVMStrategy.OVO or SVMStrategy.OVR.
        """
        shape = getattr(self._target, 'decision_function_shape', 'ovr')
        return SVMStrategy.OVO if shape == 'ovo' else SVMStrategy.OVR

    @property
    def n_support_vectors(self) -> int:
        return self._target.support_vectors_.shape[0]

    @property
    def support_vectors(self) -> np.ndarray:
        """Shape: (n_SV, n_features), dtype float32."""
        return self._target.support_vectors_.astype(np.float32)

    @property
    def coefficients(self) -> np.ndarray:
        """
        Dual coefficient matrix reconstructed from sklearn's dual_coef_.

        sklearn always stores dual_coef_ with shape (n_classes-1, n_SV),
        grouping columns by class. This method reconstructs the per-pair
        or per-class matrix that EmbedIA's C code expects.

        OvO → shape (n_pairs,   n_SV): coef[pair, sv] != 0 only if sv
              participates in that pair.
        OvR → shape (n_classes, n_SV): coef[class, sv] for all SVs.
        """
        if self.strategy == SVMStrategy.OVO:
            return self._ovo_coefficients()
        return self._ovr_coefficients()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_support_vector_labels(self) -> np.ndarray:
        """Class label (0-based index) for each support vector."""
        n_support = self._target.n_support_
        return np.concatenate([
            np.full(n_support[i], i)
            for i in range(self.n_classes)
        ]).astype(np.uint16)

    def _ovo_coefficients(self) -> np.ndarray:
        """
        Reconstruct per-pair coefficients from sklearn's dual_coef_.

        For pair (i, j):
          - SVs of class i → row j-1 of dual_coef_
          - SVs of class j → row i   of dual_coef_

        Returns float32 array of shape (n_pairs, n_SV).
        """
        n_classes = self.n_classes
        n_SV      = self.n_support_vectors
        n_pairs   = n_classes * (n_classes - 1) // 2
        dual_coef = self._target.dual_coef_       # (n_classes-1, n_SV)
        n_support = self._target.n_support_

        assert dual_coef.shape == (n_classes - 1, n_SV), (
            f"Unexpected dual_coef_ shape {dual_coef.shape}, "
            f"expected ({n_classes - 1}, {n_SV})"
        )

        coef_matrix = np.zeros((n_pairs, n_SV), dtype=np.float32)
        sv_start    = np.concatenate([[0], np.cumsum(n_support)])

        pair_idx = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                sv_i = slice(sv_start[i], sv_start[i + 1])
                sv_j = slice(sv_start[j], sv_start[j + 1])
                coef_matrix[pair_idx, sv_i] = dual_coef[j - 1, sv_i]
                coef_matrix[pair_idx, sv_j] = dual_coef[i,     sv_j]
                pair_idx += 1

        return coef_matrix

    def _ovr_coefficients(self) -> np.ndarray:
        """
        Reconstruct per-class coefficients from sklearn's dual_coef_ for OvR.

        For sklearn's OvR implementation:
        - dual_coef_ has shape (n_classes-1, n_SV)
        - Row i corresponds to classifier for class i+1
        - Values are already signed: positive for SVs of class i+1,
          negative for SVs of other classes

        Returns array of shape (n_classes, n_SV) where row i contains
        coefficients for the binary classifier that separates class i
        from the rest.
        """
        n_classes = self.n_classes
        n_SV = self.n_support_vectors
        dual_coef = self._target.dual_coef_  # (n_classes-1, n_SV)

        # Para OvR, simplemente reorganizamos:
        # La clase 0 no tiene fila propia en dual_coef_, sus coeficientes
        # están implícitamente como el negativo de la suma de los otros?
        # ¡NO! sklearn NO almacena coeficientes para la primera clase.

        # En realidad, para cada clasificador binario (clase i vs resto):
        # - Los SVs de clase i tienen coeficientes positivos en dual_coef_[i-1]
        # - Los SVs de otras clases tienen coeficientes negativos

        coef_matrix = np.zeros((n_classes, n_SV), dtype=np.float32)

        # Para clase 0: se infiere de los otros clasificadores
        # Los coeficientes de clase 0 son el negativo de la suma de todos
        # los clasificadores donde clase 0 es la clase negativa

        for i in range(1, n_classes):
            # Clasificador para clase i (i > 0)
            coef_matrix[i] = dual_coef[i - 1]

        # Clase 0: la suma de todos los coeficientes debe ser 0
        # por lo tanto: coef_matrix[0] = -sum(coef_matrix[1:])
        coef_matrix[0] = -np.sum(coef_matrix[1:], axis=0)

        return coef_matrix

    def export_debug_info(self, filename: str = "svm_debug.txt"):
        """Export model parameters to a text file for debugging."""
        with open(filename, 'w') as f:
            f.write("=== SVM Model Debug Info ===\n\n")

            f.write(f"Strategy : {self.strategy}\n")
            f.write(f"Kernel   : {self.kernel_type}\n")
            gamma, intercept, degree = self.kernel_params
            f.write(f"Params   : gamma={gamma}, intercept={intercept}, degree={degree}\n\n")

            f.write(f"Support Vectors ({self.n_support_vectors} x {self.n_features}):\n")
            for i, sv in enumerate(self.support_vectors):
                f.write(f"  SV[{i:3d}]: [{', '.join(f'{v:.6f}' for v in sv)}]\n")

            f.write(f"\nIntercepts ({len(self.intercepts)}):\n")
            for i, b in enumerate(self.intercepts):
                f.write(f"  [{i}]: {b:.6f}\n")

            coefs = self.coefficients
            f.write(f"\nCoefficients {coefs.shape}:\n")
            for idx in range(coefs.shape[0]):
                nonzero = np.where(np.abs(coefs[idx]) > 1e-6)[0]
                f.write(f"  [{idx}]: ")
                f.write(", ".join(f"SV{sv}={coefs[idx, sv]:.6f}" for sv in nonzero))
                f.write("\n")

        print(f"Debug info exported to {filename}")


class SKLSvmLinearWrapper(SVMDirectWrapperBase):
    """
    Wraps sklearn's LinearSVC for EmbedIA inference code generation.

    LinearSVC learns weight vectors directly without support vectors.
    Inference is a plain dot product per class — no kernel evaluation.
    Always uses OvR strategy.
    """

    # ─────────────────────────────────────────────────────────────────────
    # ClassifierWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        return len(self._target.classes_)

    @property
    def n_features(self) -> int:
        return self._target.n_features_in_

    # ─────────────────────────────────────────────────────────────────────
    # SVMWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def intercepts(self) -> np.ndarray:
        """Shape: (n_classes,), dtype float32."""
        return self._target.intercept_.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # SVMDirectWrapperBase
    # ─────────────────────────────────────────────────────────────────────

    @property
    def coefficients(self) -> np.ndarray:
        """
        Weight matrix learned directly during training.
        Shape: (n_classes, n_features), dtype float32.
        """
        return self._target.coef_.astype(np.float32)