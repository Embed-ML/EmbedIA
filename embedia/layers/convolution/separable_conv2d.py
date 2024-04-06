from embedia.core.layer import Layer
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class SeparableConv2D(Layer):
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
     rule is recomended: LayerClassName+"_layer". Example for SeparableConv2D class
     should be named separable_conv2d_layer.
     The invoke function receives an input and an output parameter with the parameter's
     name that are used in the predict function of the model.

     The SeparableConv2D convolutional layer is a layer that requires additional data structure
     (weights to be initialized) in addition to the input data. For this reason
     sets "_use_data_structure" to True. Because ot this, code generator generates C code
     automatically based on the content of the properties that store c code:
     - struct_data_type [automatic named]: name of data type of structure to store parameters
       like filters, kernel size, padding, etc. This structure must be declared in some .h file.
       Example: for Classname+"_layer_t" generates separable_conv2d_layer_t
     - variable_declaration [automatic generated]: variable declaration to store parameters.
       Example: for Classname+"_layer_t" LayerName+"_data" generates separable_conv2d_layer_t separable_conv2d_0_data
     - function_prototype [automatic generated]: function prototype to invoke on data initialization.
       Example: for struct_data_type "init_"+LayerName+"_data"(void)' generates
       separable_conv2d_layer_t init_separable_conv2d_data(void)
     - variable_initialization [automatic generated]: code to initialize structure variable via
       initialization function. Example: for LayerName+"_data" = "init_"+LayerName+"_data(void)"
       generates conv2d_0_data = init_separable_conv2d_0_data(void).
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
        # the type defined in "struct_data_type" must exists in "embedia.h"
        # self.struct_data_type = self.get_type_name().lower()+'_layer_t'

        self._use_data_structure = True  # this layer require data structure initialization

        # self.depth_weights = self._adapt_weights(wrapper.get_weights()[0])
        # self.point_weights = self._adapt_weights(wrapper.get_weights()[1])
        # self.biases = wrapper.get_weights()[2]


    # def _adapt_weights(self, weights):
    #     _row, _col, _can, _filt = weights.shape
    #     arr = np.zeros((_filt, _can, _row, _col))
    #     for row, elem in enumerate(weights):
    #         for col, elem2 in enumerate(elem):
    #             for chn, elem3 in enumerate(elem2):
    #                 for filt, value in enumerate(elem3):
    #                     arr[filt, chn, row, col] = value
    #     return arr

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # estimate amount multiplication and addition operations
        out_size = self.output_size

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self._wrapper.depth_weights.shape
        MACs = out_size*n_cols*n_rows*n_channels

        n_channels, n_filters, n_rows, n_cols = self._wrapper.point_weights.shape
        MACs += out_size*n_cols*n_rows*n_channels

        return MACs

    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self._wrapper.depth_weights.shape
        depth_params = n_channels * n_filters * n_rows * n_cols

        n_channels, n_filters, n_rows, n_cols = self._wrapper.point_weights.shape
        point_params = n_channels * n_filters * n_rows * n_cols

        # EmbedIA filter structure size
        sz_filter_t = 4 # 'filter_t'

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)

        mem_size = ((depth_params + point_params + n_filters) * dt_size / 8 +
                    sz_filter_t * n_filters)

        return mem_size

    @property
    def function_implementation(self):
        depth_filters, depth_channels, depth_rows, depth_columns = self._wrapper.depth_weights.shape  # Getting layer info from it's weights

        depth_kernel_size = f'{{{depth_rows}, {depth_columns}}}'  # Defining kernel size

        point_filters, point_channels, point_rows, point_cols = self._wrapper.point_weights.shape  # Getting layer info from it's weights
        point_kernel_size = f'{{{point_rows}, {point_cols}}}'

        # padding
        padding = self._wrapper.padding

        # strides
        (strd_rows, strd_cols) = (self._wrapper.strides[-2], self._wrapper.strides[-1])
        assert strd_rows == strd_cols  # only supports equal length strides in the row and column dimensions
        strides = f'{{{strd_rows}, {strd_cols}}}'

        struct_type = self.struct_data_type

        (data_type, data_converter) = self.model.get_type_converter()

        data_converter.fit(np.concatenate((self._wrapper.depth_weights.ravel(), self._wrapper.point_weights.ravel())))
        conv_depth_weights = data_converter.transform(self._wrapper.depth_weights)
        conv_point_weights = data_converter.transform(self._wrapper.point_weights)
        conv_biases = data_converter.transform(self._wrapper.biases)

        if self.is_data_quantized:
            qparams = f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'
        else:
            qparams = ''

        comm_values = self.options.data_type != ModelDataType.FLOAT # add original values as comment?
        identation = ' ' * 12
        init_conv_layer = f'''

{struct_type} init_{self.name}_data(void){{

        '''
        o_weights = '\n'
        for ch in range(depth_channels):
            for r in range(depth_rows):
                o_weights += identation
                for c in range(depth_columns):
                    o_weights += f'''{conv_depth_weights[0,ch,r,c]}, '''
                if comm_values:
                    o_weights += f'/* {self._wrapper.depth_weights[0, ch, r, 0:depth_columns]} */'
                o_weights += '\n'

        id = o_weights.rfind(',')
        o_weights = o_weights[0:id] + o_weights[id + 1:]  # remove last comma

        o_code = f'''
        static {data_type} depth_weights[]={{{o_weights}
        }};
        // static filter_t depth_filter = {{{depth_channels}, {depth_kernel_size}, depth_weights }};
        static filter_t depth_filter = {{ depth_weights }};

        static filter_t point_filters[{point_filters}];
        '''
        init_conv_layer += o_code

        for i in range(point_filters):
            o_weights = ""
            for ch in range(point_channels):
                o_weights+= f'''{conv_point_weights[i,ch,0,0]}, '''
            # o_weights = o_weights[0:-2] # remove las comma
            if comm_values:
                comm_weights = f' /* {self._wrapper.point_weights[i, ch, 0, 0:point_channels]} */'
                comm_bias = f' /* {self._wrapper.biases[i]} */'
            else:
                comm_weights = ''
                comm_bias = ''

            o_code = f'''
        static {data_type} point_weights{i}[]={{{o_weights}{comm_weights}
        }};
        static filter_t point_filter{i} = {{point_weights{i}, {conv_biases[i]}{comm_bias}}};
        point_filters[{i}] = point_filter{i};
        '''
            init_conv_layer += o_code

        init_conv_layer += f'''
        {struct_type} layer = {{{point_filters}, point_filters, {point_channels}, {point_kernel_size}, depth_filter, {depth_channels}, {depth_kernel_size}, {padding}, {strides}{qparams} }};
        return layer;
}}
        '''

        return init_conv_layer

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
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
        return f'''separable_conv2d_layer({self.name}_data, {input_name}, &{output_name});'''