
from embedia.core.layer import Layer
from embedia.model_generator.project_options import ModelDataType

import numpy as np


class Conv2D(Layer):
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


    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # layer dimensions
        n_filters, n_channels, n_rows, n_cols = self._wrapper.weights.shape

        # estimate amount multiplication and addition operations
        out_size = self.output_size
        # MACs =  (n_rows * n_cols *  n_filters) * in_size
        MACs = out_size*n_cols*n_rows*n_channels

        return MACs

    def calculate_memory(self, types_dict):
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
        sz_filter_t = types_dict['filter_t']

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)

        mem_size = (n_channels * n_rows * n_cols *
                    dt_size / 8 + sz_filter_t) * n_filters

        return mem_size


    @property
    def function_implementation(self):
        """
        Generate C code with the initialization function of the additional
        structure (defined in "embedia.h") required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """

        (data_type, data_converter) = self.model.get_type_converter()

        conv_weights = data_converter.fit_transform(self._wrapper.weights)
        conv_biases = data_converter.transform(self._wrapper.biases)
        padding = f'%d' % self._wrapper.padding
        strides = f'{{%d, %d}}' % self._wrapper.strides

        if self.is_data_quantized:
            qparams = f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'
        else:
            qparams = ''

        comm_values = self.options.data_type != ModelDataType.FLOAT # add original values as comment?

        n_filters, n_channels, n_rows, n_cols = self._wrapper.weights.shape

        kernel_size = f'{{ {n_rows}, {n_cols} }}' # Defining kernel size

        identation = ' '*8
        ret = ""
        struct_type = self.struct_data_type  # autogenerated name: conv2d_datat
        name = self.name+'_data'
        text = f'''

{struct_type} init_{name}(void){{

        static filter_t filters[{n_filters}];
        '''
        for i in range(n_filters):
            o_weights = '\n'
            for ch in range(n_channels):
                for r in range(n_rows):
                    o_weights += identation
                    for c in range(n_cols):
                        o_weights += f'   {conv_weights[i, ch, r, c]}, '
                    if comm_values:
                        o_weights += f'/* {self._wrapper.weights[i, ch, r, 0:n_cols]} */'
                    o_weights += '\n'

            id = o_weights.rfind(',')
            o_weights = o_weights[0:id] + o_weights[id+1:]  # remove last comma

            if comm_values:
                bias_weight = f' //{self._wrapper.biases[i]}'
            else:
                bias_weight = ''

            o_code = f'''
        static const {data_type} weights{i}[]={{{o_weights}        }};
        static filter_t filter{i} = {{ weights{i}, {conv_biases[i]}}}; {bias_weight}
        filters[{i}]=filter{i};
            '''
            text += o_code
        text += f'''
        conv2d_layer_t layer = {{{n_filters}, filters, {n_channels}, {kernel_size}, {padding}, {strides}{qparams} }};
        return layer;
}}
        '''
        ret += text

        return ret

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Conv2D class associated to the Keras
        Conv2D layer, the function "conv2d_layer" must be implemented in
        "embedia.c"

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
        # change function name for some optimizations
        if self._wrapper.padding == 1: #same
            opt_name = '_padding'
        elif self._wrapper.strides[0]>1 or self._wrapper.strides[1]>1:
            opt_name = '_strides'
        else:
            opt_name = ''
        return f'''conv2d{opt_name}_layer({self.name}_data, {input_name}, &{output_name});'''
