from embedia.layers.data_layer import DataLayer
from embedia.model_generator.project_options import ModelDataType
from embedia.utils import file_management
import numpy as np


class DepthwiseConv2D(DataLayer):

    def __init__(self, model, layer, options, **kwargs):

        super().__init__(model, layer, options, **kwargs)
        # the type defined in "struct_data_type" must exists in "embedia.h"
        # self.struct_data_type = self.get_type_name().lower()+'_layer_t'
        w = layer.get_weights()
        self.weights = self._adapt_weights(w[0])
        self.biases = w[1]

    def _adapt_weights(self, weights):
        _row, _col, _can, _filt = weights.shape
        arr = np.zeros((_filt, _can, _row, _col))
        for row, elem in enumerate(weights):
            for col, elem2 in enumerate(elem):
                for chn, elem3 in enumerate(elem2):
                    for filt, value in enumerate(elem3):
                        arr[filt, chn, row, col] = value
        return arr

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # estimate amount multiplication and addition operations
        out_size = self.get_output_size()

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self.weights.shape
        MACs = out_size*n_cols*n_rows*n_channels

        #n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        #MACs += out_size*n_cols*n_rows*n_channels

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
        n_channels, n_filters, n_rows, n_cols = self.weights.shape
        depth_params = n_channels * n_filters * n_rows * n_cols

        #n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        #point_params = n_channels * n_filters * n_rows * n_cols

        # EmbedIA filter structure size
        sz_filter_t = types_dict['filter_t']

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)
        if self.options.data_type == ModelDataType.BINARY:
            dt_size = 32

        mem_size = ((depth_params + n_filters) * dt_size / 8 +
                    sz_filter_t * n_filters)

        return mem_size

    def functions_init(self):

        (data_type, data_converter) = self.model.get_type_converter()

        # data_converter.fit(np.concatenate((self.weights.ravel(), self.biases.ravel())))
        #
        # conv_weights = data_converter.transform(self.weights)
        # conv_biases = data_converter.transform(self.biases)

        qparams = ''

        conv_weights = data_converter.fit_transform(self.weights)
        if self.is_data_quantized():
            qparams += f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'
        conv_biases = data_converter.fit_transform(self.biases)
        if self.is_data_quantized():
            qparams += f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'


        # add original comment values
        comm_values = self.options.data_type != ModelDataType.FLOAT

        depth_filters, depth_channels, depth_rows, depth_columns = self.weights.shape  # Getting layer info from it's weights
        assert depth_rows == depth_columns  # WORKING WITH SQUARE KERNELS FOR NOW
        depth_kernel_size = depth_rows  # Defining kernel size

        # point_filters, point_channels, _, _ = self.point_weights.shape  # Getting layer info from it's weights

        struct_type = self.struct_data_type

        comm_values = self.options.data_type != ModelDataType.FLOAT  # add original values as comment?
        identation = ' '*8

        init_conv_layer = f'''
{struct_type} init_{self.name}_data(void){{
'''
        d_weights = ''
        for ch in range(depth_channels):
            for r in range(depth_rows):
                d_weights += '\n' + identation
                for c in range(depth_columns):
                    d_weights += f'''{conv_weights[0,ch,r,c]}, '''
                if comm_values:
                    d_weights += f'/* {self.weights[0, ch, r, 0:depth_columns]} */'
            d_weights += '\n'

        if comm_values:
            id = d_weights.rfind(',')
            d_weights = d_weights[0:id] + d_weights[id+1:] # remove last comma

        b_weights = '\n'
        for ch in range(depth_channels):
            b_weights += identation + f'{conv_biases[ch]}, '
            if comm_values:
                b_weights += f'/* {self.biases[ch]} */'
            b_weights += '\n'
        id = b_weights.rfind(',')
        b_weights = b_weights[0:id] + b_weights[id+1:] # remove last comma

        o_code = f'''
    static const {data_type} weights[]={{{d_weights}    }};
    static const {data_type} biases[]={{{b_weights}    }};
'''
        init_conv_layer += o_code

        # analizar lo que sigue
        init_conv_layer += f'''
    {struct_type} layer = {{{depth_channels}, {depth_kernel_size}, weights, biases{qparams} }};
        
    return layer;
}}
'''

        return init_conv_layer

    def predict(self, input_name, output_name):
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

        return f'''depthwise_conv2d_layer({self.name}_data, {input_name}, &{output_name});'''