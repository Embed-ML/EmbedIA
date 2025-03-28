from embedia.core.neural_net_layer import NeuralNetLayer


class ZeroPadding2D(NeuralNetLayer):
    """
    The ZeroPadding2D layer is a layer that does not require additional data beyond
    the input data (have padding size, but are parameters of the related function.
    For this reason it inherits from the "Layer" class that
    implements the basic behavior of an EmbedIA layer/element.
    Normally, the programmer must implement the "predict" method, with the
    invocation to the EmbedIA function (previously implemented in "neural_net.c")
    which performs the processing of the layer.

    Layer wrapper required properties:
        - padding => (height, width)
    """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        #self.input_data_type = f'data3d_t'
        #self.output_data_type = 'data3d_t'

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name"+"input dimension" + "d_layer".
        For example, for the EmbedIA Flatten class associated to the Keras
        Flatten layer with an input size of 3, the function "flatten3d_layer"
        must be implemented in "neural_net.c"

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "neural_net.c".

        """

        (pad_h, pad_w) = (self.wrapper.padding[0], self.wrapper.padding[1])
        return f'''zero_padding2d_layer({pad_h}, {pad_w}, {input_name}, &{output_name});'''
