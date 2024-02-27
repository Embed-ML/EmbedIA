from embedia.layers.data_layer import Layer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class ZeroPadding2D(Layer):
    """
    The ZeroPadding2D layer is a layer that does not require additional data beyond
    the input data (have padding size, but are parameters of the related function.
    For this reason it inherits from the "Layer" class that
    implements the basic behavior of an EmbedIA layer/element.
    Normally, the programmer must implement the "predict" method, with the
    invocation to the EmbedIA function (previously implemented in "embedia.c")
    which performs the processing of the layer.
    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)

        self.input_data_type = f'data3d_t'
        self.output_data_type = 'data3d_t'

    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name"+"input dimension" + "d_layer".
        For example, for the EmbedIA Flatten class associated to the Keras
        Flatten layer with an input size of 3, the function "flatten3d_layer"
        must be implemented in "embedia.c"

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
            processing of the layer in the file "embedia.c".

        """

        (pad_h, pad_w) = (self.layer.padding[0], self.layer.padding[1])
        return f'''zero_padding2d_layer({pad_h}, {pad_w}, {input_name}, &{output_name});'''
