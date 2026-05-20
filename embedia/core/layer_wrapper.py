from abc import abstractmethod
from enum import Enum


class OutputPredictionType(Enum):
    """Defines how model outputs should be processed for final prediction. This classification determines
    the post-processing steps required to transform raw model outputs into meaningful predictions. The type
    is automatically inferred from the model's last layer configuration.
    Values:
        DIRECT_CLASS_ID (0): Model directly outputs class indices (e.g., sklearn trees). No argmax or threshold needed.
        CLASS_PROBABILITIES (1): Model outputs probability distributions (e.g., softmax). Requires argmax for class prediction
        BINARY_OUTPUT (2): Single floating-point output (e.g., sigmoid). Requires threshold comparison (default: 0.5).
        REGRESSION_OUTPUT (3): Continuous output values (single or multiple). No processing needed.
    """
    DIRECT_CLASS_ID = 0     # Model returns class index directly (no processing)
    CLASS_PROBABILITIES = 1 # Model returns probabilities (needs argmax)
    BINARY_OUTPUT = 2       # Single float output (needs threshold)
    REGRESSION_OUTPUT = 3   # Single or multiple regression values



class LayerWrapper:

    def __init__(self, target: object):
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def name(self):
        if hasattr(self._target, 'name'):
            return self._target.name
        if hasattr(self._target, '__name__'):
            return self._target.__name__
        if hasattr(self._target, '__class__'):
            return self._target.__class__.__name__
        return self.__class__.__name__.removesuffix('Wrapper')

    @property
    def input_shape(self):
        return None

    @property
    def output_shape(self):
        return None

    @property
    def output_prediction_type(self):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Classifier contract
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierWrapperBase(LayerWrapper):
    """
    Base contract for all classifiers.

    Any classifier wrapper must provide these properties so that the code
    generator can build the C project without knowing the underlying library.
    """

    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        """Number of input features."""
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple:
        """Expected input shape: (None, n_features)."""
        return (None, self.n_features)

    @property
    def output_shape(self) -> tuple:
        """Expected output shape: (None, n_classes)."""
        return (None, self.n_classes)

    @property
    def output_prediction_type(self) -> OutputPredictionType:
        """How to interpret the raw output to get a class prediction."""
        raise NotImplementedError


