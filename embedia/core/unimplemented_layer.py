from embedia.core.layer import Layer
from embedia.core.exceptions import UnsupportedLayerError
from embedia.model_generator.project_options import UnimplementedLayerAction
from tensorflow.keras.layers import Layer as TFKLayer


class UnimplementedLayer(Layer):
    """
    This class defines the behavior of an EmdedIA layer/element that has not
    been implemented. Sometimes the lack of this element may be considered an
    error, sometimes it may be forced to be ignored and sometimes it may be
    ignored because it is known to play no role in inference such as the
    'Random' layers of Tensorflow/Keras
    """

    def __init__(self, model, wrapper, **kwargs):
        """
        Constructor that receives:
            - EmbedIA Model
            - object (Keras, SkLearn, etc.) associated to the EmbedIA layer
            - EmbedIA project options
        Parameters
        ----------
        wrapper : object
            layer/object is associated to this EmbedIA layer/element. For
            example, it can receive a Keras layer or a SkLearn scaler.
        Returns
        -------
        None.
        """
        super().__init__(model, wrapper, **kwargs)

        self.resolve()

    def resolve(self):
        if self.options is None or self.options.on_unimplemented_layer == UnimplementedLayerAction.FAILURE:
            raise UnsupportedLayerError(self.wrapper)

        if self.options.on_unimplemented_layer == UnimplementedLayerAction.IGNORE_ALL:
            self.message = 'Not implemented. Forced to ignore'
        elif self.options.on_unimplemented_layer == UnimplementedLayerAction.IGNORE_KNOWN:
            if self.is_known_layer():
                self.message = 'Not implemented. Required in training but not in prediction'
            else:
                raise UnsupportedLayerError(self.wrapper)

    def is_known_layer(self):
        known = (isinstance(self.wrapper, TFKLayer)
                 and self.wrapper.__class__.__name__.startswith('Random'))

        return known

    def invoke(self, input_name, output_name):
        return  '// '+self.message

    def activation_function(self, param):
        # Nothing to do
        return ''

    def debug_function(self, param):
        # Nothing to do
        return ''
