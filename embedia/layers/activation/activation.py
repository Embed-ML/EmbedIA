from embedia.core.layer import Layer
from embedia.layers.activation.activation_functions import ActivationFunctions
from tensorflow.keras.layers import Activation as KerasActivation


class Activation(Layer):
    """
    Normally all layers can directly incorporate activation functions. However,
    sometimes this functionality can appear as an independent layer. The EmbedIA
    activation layer is associated with the Keras Activation layer/object.

    Default Tensorflow wrapper properties are required (name, activation, input_shape, output_shape)

    """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        # saves output into output of previous layer
        self._inplace_output = True

        # # layer can be a Keras layer with activation or an Activation layer
        # if not isinstance(wrapper.target, KerasActivation):
        #     # rename with keras layer partial name
        #     self._name = model.get_unique_name(wrapper.name + '_' + self._activation_function.get_function_name())

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be implemented in
        "embedia.c" and by convention should be called
        "class name" + "_activation" or "function name" + "_activation".
        For example, for the Keras Sigmoid Activation, the function
        "sigmoid_activation" must be implemented in "embedia.c"
        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer. Not used in activation
            functions since the output_name variable is directly modified
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.
        Returns
        -------
        str
            C code with the invocation of the activation function in the file
            "embedia.c" that performs the processing of the layer
        """
        output_size = self.output_size # number of elements number

        qparams = ''
        act_fncs = ActivationFunctions(self._model, self._wrapper)

        return act_fncs.invoke(f'{output_name}.data', output_size, qparams)

