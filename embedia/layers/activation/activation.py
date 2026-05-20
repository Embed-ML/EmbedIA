from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.layers.activation.activation_functions import ActivationFunctions
from tensorflow.keras.layers import Activation as KerasActivation


class Activation(NeuralNetLayer):
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
        self._is_dummy = wrapper.function_name == 'linear'
        # # layer can be a Keras layer with activation or an Activation layer
        # if not isinstance(wrapper.target, KerasActivation):
        #     # rename with keras layer partial name
        #     self._name = model.get_unique_name(wrapper.name + '_' + self._activation_function.get_function_name())

    def calculate_ACOPS(self):
        """
        Calculates non-MACC operations for activation layers including:
        - Comparison operations (ReLU, LeakyReLU)
        - Exponential operations (Sigmoid, Tanh)
        - Memory access operations

        Returns
        -------
        int
            Total count of non-MACC operations (ACOPS)
        """
        total_output_elements = self.output_size

        # Get activation type from wrapper
        activation_type = self.wrapper.function_name

        if activation_type:
            if activation_type in ['relu', 'leakyrelu']:
                # ReLU: 1 comparison per element (x > 0 ? x : 0)
                # LeakyReLU: 1 comparison + 1 multiplication (for negative slope)
                activation_ops = total_output_elements * (2 if activation_type == 'leaky_relu' else 1)

            elif activation_type in ['sigmoid', 'tanh']:
                # Sigmoid: 1 exp + 1 division + 1 addition (1/(1 + exp(-x)))
                # Tanh: Similar complexity to sigmoid
                activation_ops = total_output_elements * 3

            elif activation_type == 'softmax':
                # Softmax: exp + sum + division for each element
                activation_ops = total_output_elements * 3  # Simplified estimate

            else:
                activation_ops = 0

        # Memory operations (1 read + 1 write per element)
        memory_ops = 2 * total_output_elements

        return activation_ops + memory_ops

    @property
    def layer_type_name(self):
        return f'{self.__class__.__name__}({self.wrapper.function_name})'


    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be implemented in
        "neural_net.c" and by convention should be called
        "class name" + "_activation" or "function name" + "_activation".
        For example, for the Keras Sigmoid Activation, the function
        "sigmoid_activation" must be implemented in "neural_net.c"
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
            "neural_net.c" that performs the processing of the layer
        """
        output_size = self.output_size # number of elements number

        act_fncs = ActivationFunctions(self._model, self._wrapper)

        return act_fncs.invoke(f'{output_name}.data', output_size)

