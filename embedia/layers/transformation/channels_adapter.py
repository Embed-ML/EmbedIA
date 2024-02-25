from embedia.layers.layer import Layer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType

class ChannelsAdapter(Layer):

    """
    Normally the programmer must implement the method "predict",
    where the programmer must invoke the function EmbedIA function
    (implemented in "embedia.c") that should perform the layer processing.
    To avoid overlapping names, both the function name and the variable name
    are automatically generated using the layer name.
"""
    _shape = None # input/playground shape

    # Constructor receives sklearn normalization object (Scaler) in layer
    def __init__(self, model, shape, options=None, **kwargs):
        if len(shape) >= 1 and shape[0] is None:
            shape = shape[1:]
        self._shape = shape

        super().__init__(model=model, layer=None, options=options, **kwargs)


    def get_input_shape(self):
        """
        get the shape of the input of the EmbedIA layer/element.
        Returns
        -------
        n-tuple
            returns the input shape of the layer/element.
        """

        return self._shape

    def get_output_shape(self):
        """
        get the shape of the playground of the EmbedIA layer/element.
        Returns
        -------
        n-tuple
            returns the playground shape of the layer/element.
        """
        return self._shape

    def predict(self, input_name, output_name):
        return f'''channel_adapt_layer({input_name}, &{output_name});'''
