from embedia.core.neural_net_layer import NeuralNetLayer


class Flatten(NeuralNetLayer):
    """
    The Flatten layer is a layer that does not require additional data beyond
    the input data. For this reason it inherits from the "Layer" class that
    implements the basic behavior of an EmbedIA layer/element.
    Normally, the programmer must implement the "predict" method, with the
    invocation to the EmbedIA function (previously implemented in "neural_net.c")
    which performs the processing of the layer.
    """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)


        #self._output_data_type = 'data1d_t'


        #dims = len(self.input_shape)
        # define C data types of input/output data. Note that the data type of
        # the input depends on the dimensions of the input.
        #self._input_data_type = f'data{dims}d_t'
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
        dims = len(self.input_shape)
        fn_name = f'flatten{dims}d_layer'
        return f'''{fn_name}({input_name}, &{output_name});'''
