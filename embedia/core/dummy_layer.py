from embedia.core.layer import Layer

class DummyLayer(Layer):
    """
    This class defines the behavior of an EmdedIA layer/element that does not
    play any purpose in the inference process
    """

    def __init__(self, model, target, **kwargs):
        """
        Constructor that receives:
            - EmbedIA Model
            - object (Keras, SkLearn, etc.) associated to the EmbedIA layer
            - EmbedIA project options
        Parameters
        ----------
        target : object
            layer/object is associated to this EmbedIA layer/element. For
            example, it can receive a Keras layer or a SkLearn scaler.
        Returns
        -------
        None.
        """
        super().__init__(model, target, **kwargs)


    def invoke(self, input_name, output_name):
        # Nothing to do
        return ''

    def activation_function(self, param):
        # Nothing to do
        return ''

    def debug_function(self, param):
        # Nothing to do
        return ''
