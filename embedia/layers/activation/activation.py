from embedia.layers.layer import Layer


class Activation(Layer):
    """
    Normally all layers can directly incorporate activation functions. However,
    sometimes this functionality can appear as a independent layer. The
    EmbedIA activation layer is associated with the Keras Activation layer. In
    particular this class does not implement the "predict" method because the
    invocation to the EmbedIA activation function is resolved when the code
    generator invokes the "activation_function()" method as it happens with
    other layers.
    """

    def __init__(self, model, layer, options=None, **kwargs):
        super().__init__(model, layer, options, **kwargs)
        self.input_data_type = "data1d_t"
        self.output_data_type = "data1d_t"

        # saves output into output of previous layer
        self.inplace_output = True
