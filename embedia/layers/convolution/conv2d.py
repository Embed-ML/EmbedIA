
from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.core.padding_types import PaddingType
from embedia.model_generator.project_options import ModelDataType

import numpy as np


class Conv2D(NeuralNetLayer):
    """

    Develop info:
    This class must define the behavior of an EmdedIA layer/element. It defines
    methods/properties to obtain information related to the inputs and outputs of
    the layer/element such as its shape, number of elements, EmbedIA associated
    data type.
    It also implements methods to generate the C code necessary for debugging
    function and invocation of the C function associated to the layer/element.
    This function must be implemented in some .c file its prototype declared
    in respective .h. The name of function can be anything but an EmbedIA naming
    rule is recomended: LayerClassName+"_layer". Example for Conv2D class should
    be named conv2d_layer.
    The invoke function receives an input and an output parameter with the parameter's
    name that are used in the predict function of the model.

    The Conv2D convolutional layer is a layer that requires additional data structure
    (weights to be initialized) in addition to the input data. For this reason
    sets "_use_data_structure" to True. Because ot this, code generator generates C code
    automatically based on the content of the properties that store c code:
    - struct_data_type [automatic named]: name of data type of structure to store parameters
      like filters, kernel size, padding, etc. This structure must be declared in some .h file.
      Example: for Classname+"_layer_t" generates conv2d_layer_t
    - variable_declaration [automatic generated]: variable declaration to store parameters.
      Example: for Classname+"_layer_t" LayerName+"_data" generates conv2d_layer_t conv2d_0_data
    - function_prototype [automatic generated]: function prototype to invoke on data initialization.
      Example: for struct_data_type "init_"+LayerName+"_data"(void)' generates
      conv2d_layer_t init_conv2d_data(void)
    - variable_initialization [automatic generated]: code to initialize structure variable via
      initialization function. Example: for LayerName+"_data" = "init_"+LayerName+"_data(void)"
      generates conv2d_0_data = init_conv2d_0_data(void).
    - function_implementation [user generated]: full code of initialization function. User must
      generate code to initialize the data structure.

    Layer wrapper required properties:
        - padding => 0=valid, 1=same
        - strides => (height, width)
        - weights => 4d array formatted: filters, channel, row, column
        - biases => 1d array

   """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization


    def calculate_params(self):
        """
        Calculates the number of trainable and non-trainable parameters in a 2D convolutional layer.

        Returns
        -------
        tuple
            (number_of_trainable_params, number_of_non_trainable_params)
        """

        # A standard Conv2D layer has only trainable parameters: kernel weights + biases

        # Shape convention: (num_filters, num_channels, kernel_height, kernel_width)
        n_filters, n_channels, n_rows, n_cols = self.wrapper.weights.shape

        # Kernel parameters: each filter has (n_channels × kernel_height × kernel_width) weights
        # Total across all filters: n_filters × n_channels × n_rows × n_cols
        trainable_kernels = n_filters * n_channels * n_rows * n_cols

        trainable_biases = n_filters # Bias parameters: one bias per output filter

        total_trainable = trainable_kernels + trainable_biases
        total_non_trainable = 0 # Standard Conv2D layers have no non-trainable parameters

        return (total_trainable, total_non_trainable)
    #
    # def calculate_MAC(self):
    #     """
    #     calculates amount of multiplication and accumulation operations
    #     Returns
    #     -------
    #     int
    #         amount of multiplication and accumulation operations
    #
    #     """
    #     # layer dimensions
    #     n_filters, n_channels, n_rows, n_cols = self._wrapper.weights.shape
    #
    #     # estimate amount multiplication and addition operations
    #     out_size = self.output_size
    #     # MACs =  (n_rows * n_cols *  n_filters) * in_size
    #     MACs = out_size*n_cols*n_rows*n_channels
    #
    #     return MACs

    def calculate_MAC(self):
        """
        Calculates the number of Multiply-Accumulate (MAC) operations in a 2D convolutional layer.

        For a conv layer, each output element requires:
        - n_channels * kernel_height * kernel_width multiplications
        - (n_channels * kernel_height * kernel_width - 1) additions
        (Each multiply-add pair counts as 1 MAC)

        Returns
        -------
        int
            Total number of MAC operations for the layer
        """
        # Shape: (num_filters, num_channels, kernel_height, kernel_width)
        n_filters, n_channels, n_rows, n_cols = self._wrapper.weights.shape

        # Calculate total MAC operations:
        # = num_output_pixels * operations_per_pixel
        # = (out_h * out_w * n_filters) * (n_channels * kernel_h * kernel_w)
        total_MACs = self.output_size * n_filters * n_channels * n_rows * n_cols

        return total_MACs

    def calculate_ACOPS(self):
        """
        Calculates the number of non-MACC operations (ACOPS) in a Conv2D layer:
        - Bias additions (arithmetic)
        - Memory access operations (load/store)

        Returns
        -------
        int
            Total count of non-MACC operations (ACOPS)
        """
        # Shape: (num_filters, num_channels, kernel_height, kernel_width)
        n_filters, _, _, _ = self.wrapper.weights.shape

        output_size = self.output_size  # Get output feature map dimensions (height * width)

        # Bias additions: 1 addition per output element (if bias exists)
        bias_ops = (output_size * n_filters) * len(self.wrapper.biases)

        # Memory operations: minimum 2 memory ops per output element: write output + read input patch
        memory_ops = 2 * output_size * n_filters

        return bias_ops  + memory_ops # Total ACOPS = sum of all non-MACC operations

    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        n_filters, n_channels, n_rows, n_cols = self._wrapper.weights.shape

        # EmbedIA filter structure size
        # struct { float * weights; float bias; }filter_t;

        sz_filter_t = 4

        # base data type in bits: float, fixed (32/16/8)
        dt_size = self.options.data_type.size

        mem_size = (n_channels * n_rows * n_cols *
                    dt_size / 8 + sz_filter_t) * (n_filters+1)

        return mem_size

    @property
    def function_implementation(self):
        """
        Generate C code for the conv2d layer initialization function.
        Uses CBuilder.add_array and add_struct for clean, declarative generation.
        """
        weights = self._wrapper.weights
        biases = self._wrapper.biases
        padding = '%d' % self._wrapper.padding
        strides = '{%d, %d}' % self._wrapper.strides
        n_filters, n_channels, n_rows, n_cols = weights.shape

        (weight_type, weight_converter) = self.model.get_type_converter()
        conv_weights = weight_converter.fit_transform(weights)

        if self.options.data_type == ModelDataType.QUANT8:
            (_, biases_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            conv_biases = biases_converter.transform(biases)
            params = weight_converter.export_params(mode='q15')
            qparams = f', {{ {params.scale_q}, {params.zero_point} }}'
        else:
            (_, biases_converter) = self.model.get_type_converter()
            conv_biases = biases_converter.transform(biases)
            qparams = ''

        comm_values = self.options.data_type != ModelDataType.FLOAT
        kernel_size = f'{{ {n_rows}, {n_cols} }}'
        name = self.name + '_data'
        cb = self.c_builder

        with cb.bgn(f'\n{self.struct_data_type} init_{name}(void)'):

            # --- weight arrays: one per filter, laid out as rows of n_cols values ---
            for i in range(n_filters):
                flat_weights = conv_weights[i].flatten().tolist()
                row_comments = None
                if comm_values:
                    row_comments = [
                        str(weights[i, ch, r, 0:n_cols])
                        for ch in range(n_channels)
                        for r in range(n_rows)
                    ]
                cb.add_array(
                    f'static EMBEDIA_MODEL_STORAGE {weight_type}',
                    f'weights{i}',
                    flat_weights,
                    cols=n_cols,
                    comments=row_comments
                )

            cb.add('')

            # --- filters array: one entry per filter with weights pointer + bias ---
            filter_inits = [
                f'{{ weights{i}, {conv_biases[i]} }}'
                for i in range(n_filters)
            ]
            bias_comments = [str(biases[i]) for i in range(n_filters)] if comm_values else None
            cb.add_array(
                'static EMBEDIA_MODEL_STORAGE filter_t',
                'filters',
                filter_inits,
                comments=bias_comments
            )

            cb.add('')

            # --- layer struct ---
            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {self.struct_data_type}',
                'layer',
                [str(n_filters), 'filters', str(n_channels),
                 kernel_size, padding, strides + qparams]
            )

            cb.add('')
            cb.add('return layer;')

        return cb.get_code()


    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Conv2D class associated to the Keras
        Conv2D layer, the function "conv2d_layer" must be implemented in
        "neural_net.c"

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
        # change function name for some optimizations
        if self._wrapper.padding == PaddingType.SAME:
            opt_name = '_padding'
        elif self._wrapper.strides[0]>1 or self._wrapper.strides[1]>1:
            opt_name = '_strides'
        else:
            opt_name = ''
        return f'''conv2d{opt_name}_layer({self.name}_data, {input_name}, &{output_name});'''
