from embedia.core.layer import Layer
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class DepthwiseConv2D(Layer):

    def __init__(self, model, target, **kwargs):

        super().__init__(model, target, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

        w = target.get_weights()
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
        out_size = self.output_size

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

    @property
    def function_implementation(self):

        (data_type, data_converter) = self.model.get_type_converter()

        # data_converter.fit(np.concatenate((self.weights.ravel(), self.biases.ravel())))
        #
        # conv_weights = data_converter.transform(self.weights)
        # conv_biases = data_converter.transform(self.biases)

        qparams = ''

        conv_weights = data_converter.fit_transform(self.weights)
        if self.is_data_quantized:
            qparams += f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'
        conv_biases = data_converter.fit_transform(self.biases)
        if self.is_data_quantized:
            qparams += f',{{ {data_converter.scale}, {data_converter.zero_pt} }}'


        # add original comment values
        comm_values = self.options.data_type != ModelDataType.FLOAT

        depth_filters, depth_channels, depth_rows, depth_columns = self.weights.shape  # Getting layer info from it's weights

        kernel_size = f'{{ {depth_rows}, {depth_columns} }}'  # Defining kernel size

        # padding
        padding = 1 if self.target.padding == 'same' else 0

        # strides
        (strd_rows, strd_cols) = (self.target.strides[-2], self.target.strides[-1])
        assert strd_rows == strd_cols  # only supports equal length strides in the row and column dimensions
        strides = f'{{{strd_rows}, {strd_cols}}}'


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
    {struct_type} layer = {{weights, biases, {depth_channels}, {kernel_size}, {padding}, {strides}{qparams} }};
        
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

        return f'''depthwise_conv2d_layer({self.name}_data, {input_name}, &{output_name});'''