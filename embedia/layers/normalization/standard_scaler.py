from embedia.layers.normalization.normalization import Normalization


class StandardNormalization(Normalization):
    """
    The StandardNormalization layer is a normalization layer that requires
    additional data (coefficients and averages to be initialized) in addition
    to the input data. For this reason it inherits the Normalization class
    which in turn inherits from DataLayer which generates C code automatically
    with the variable that will store the data, the declaration of the
    prototype of the initialization function and the call to it.
    The Normalization class implements the basic behavior of all normalizations
    that use coefficients and/or averages in their operation.
    Normally the programmer should only assign the "sub_values" properties with
    the mean values and the "div_values" property with the division
    coefficients.
    """

    def __init__(self, model, wrapper, **kwargs):

        # self.sub_values = wrapper.mean_
        # self.div_values = wrapper.scale_
        #
        # self.norm_function_name = 'standard_norm_layer'

        super().__init__(model, wrapper, **kwargs)
