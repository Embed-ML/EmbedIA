from embedia.layers.normalization.normalization import Normalization


class MaxAbsNormalization(Normalization):
    """
    The MaxAbsNormalization layer is a normalization layer that requires
    additional data (coefficients and averages to be initialized) in addition
    to the input data. For this reason it inherits the Normalization class
    which in turn inherits from DataLayer which generates C code automatically
    with the variable that will store the data, the declaration of the
    prototype of the initialization function and the call to it.
    The Normalization class implements the basic behavior of all normalizations
    that use coefficients and/or averages in their operation.
    Normally the programmer should only assign the "sub_values" properties with
    the mean values, the "div_values" property with the division
    coefficients and the "norm_function_name" property with the name of
    EmbedIA function declared in "neural_net.h"
    """

    def __init__(self, model, wrapper, **kwargs):

        super().__init__(model, wrapper, **kwargs)

