from embedia.core.layer import Layer


class ChannelsAdapter(Layer):

    """
    Normally the programmer must implement the method "predict",
    where the programmer must invoke the function EmbedIA function
    (implemented in "embedia.c") that should perform the layer processing.
    To avoid overlapping names, both the function name and the variable name
    are automatically generated using the layer name.
"""
    _shape = None # input/output shape

    # Constructor receives sklearn normalization object (Scaler) in layer
    def __init__(self, model, shape, options=None, **kwargs):
        if len(shape) >= 1 and shape[0] is None:
            shape = shape[1:]
        self._shape = shape

        super().__init__(model=model, target=None, **kwargs)


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
        get the shape of the output of the EmbedIA layer/element.
        Returns
        -------
        n-tuple
            returns the output shape of the layer/element.
        """
        return self._shape

    def invoke(self, input_name, output_name):
        return f'''channel_adapt_layer({input_name}, &{output_name});'''
