from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.core.padding_types import PaddingType
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class SeparableConv2D(NeuralNetLayer):
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
        # the type defined in "struct_data_type" must exists in "neural_net.h"
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


    def _calc_output_dim(self, in_size: int, kernel: int, stride: int, padding) -> int:
        """Calculate output dimension based on input size, kernel, stride and padding type"""
        if padding == PaddingType.VALID or padding == 'VALID':
            return max((in_size - kernel) // stride + 1, 1)
        elif padding == PaddingType.SAME or padding == 'SAME':
            return (in_size + stride - 1) // stride
        else:
            # Fallback for backward compatibility with string values
            if isinstance(padding, str):
                if padding.upper() == 'VALID':
                    return max((in_size - kernel) // stride + 1, 1)
                elif padding.upper() == 'SAME':
                    return (in_size + stride - 1) // stride
            raise ValueError(f"Padding no soportado: {padding}. Use PaddingType.VALID o PaddingType.SAME")


    @property
    def internal_alloc_required(self) -> int:
        """
        Memoria adicional que solicita internamente la capa mediante swap_alloc,
        sin contar el input ni la salida final (que son manejados externamente).

        En esta implementación, el único buffer intermedio significativo es
        la salida de depthwise, que coexiste con la salida final durante pointwise.
        """
        input_sz = self.input_shape
        kernel_sz = self._wrapper.kernel_size
        stride_sz = self._wrapper.strides
        padding = self._wrapper.padding
        h_out = self._calc_output_dim(input_sz[0], kernel_sz[0], stride_sz[0], padding)
        w_out = self._calc_output_dim(input_sz[1], kernel_sz[1], stride_sz[1], padding)

        ch, outv, outh = self.output_shape
        # Solo la salida depthwise es la alocación intermedia "extra"
        depth_multiplier = self._wrapper.depth_weights.shape[0]  # Obtener depth_multiplier del wrapper
        input_channels = self._wrapper.depth_weights.shape[1]    # Obtener input_channels del wrapper
        depth_channels = input_channels * depth_multiplier
        if self.options.data_type == ModelDataType.QUANT8:
            data_size = 16
        else:# Para datos binarios, cada valor ocupa 1 bit, así que convertimos a bytes
            data_size = self.options.data_type.size
        depth_size_bytes = depth_channels * h_out * w_out * (data_size // 8)

        return depth_size_bytes


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
        if self.options.data_type == ModelDataType.QUANT8:
            dt_size = 16
        else:
            dt_size = self.options.data_type.size

        mem_size = ((depth_params + point_params + n_filters) * dt_size / 8 +
                    sz_filter_t * n_filters)

        return mem_size

    """@property
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
"""

    @property
    def function_implementation(self):
        # Obtener dimensiones CORRECTAMENTE
        # wrapper.depth_weights es (depth_multiplier, channels, height, width)
        depth_multiplier, depth_channels, depth_rows, depth_columns = self._wrapper.depth_weights.shape

        # wrapper.point_weights es (filters, channels*depth_multiplier, 1, 1)
        point_filters, point_channels, point_rows, point_cols = self._wrapper.point_weights.shape

        # Verificar consistencia
        expected_point_channels = depth_channels * depth_multiplier
        if point_channels != expected_point_channels:
            raise ValueError(
                f"Inconsistencia: point_channels={point_channels}, pero depth_channels*depth_multiplier={depth_channels}*{depth_multiplier}={expected_point_channels}")

        depth_kernel_size = f'{{{depth_rows}, {depth_columns}}}'
        point_kernel_size = f'{{{point_rows}, {point_cols}}}'

        # padding
        padding = self._wrapper.padding

        # strides
        (strd_rows, strd_cols) = (self._wrapper.strides[-2], self._wrapper.strides[-1])
        assert strd_rows == strd_cols
        strides = f'{{{strd_rows}, {strd_cols}}}'

        struct_type = self.struct_data_type
        (data_type, data_converter) = self.model.get_type_converter()

        if self.options.data_type == ModelDataType.QUANT8:
            (biases_type, biases_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
        else:
            (biases_type, biases_converter) = self.model.get_type_converter()

        # Convertir weights
        data_converter.fit(np.concatenate((
            self._wrapper.depth_weights.ravel(),
            self._wrapper.point_weights.ravel()
        )))

        conv_depth_weights = data_converter.transform(self._wrapper.depth_weights)
        conv_point_weights = data_converter.transform(self._wrapper.point_weights)
        conv_point_biases = biases_converter.transform(self._wrapper.biases)

        # Obtener depth_biases (siempre será array de ceros de length = depth_channels * depth_multiplier)
        depth_biases = self._wrapper.depth_biases
        conv_depth_biases = biases_converter.transform(depth_biases)
        raw_depth_biases = depth_biases

        if self.is_data_quantized:
            params = data_converter.export_params(mode='q15')
            qparams = f', {{ {params.scale_q}, {params.zero_point} }}'
        else:
            qparams = ''

        comm_values = self.options.data_type != ModelDataType.FLOAT
        identation = ' ' * 8

        init_conv_layer = f'''
        {struct_type} init_{self.name}_data(void){{'''

        # Generar depth_weights - NOTA: depth_multiplier puede ser > 1
        o_weights = ''
        total_depth_filters = depth_multiplier * depth_channels

        # Para cada combinación de depth_multiplier y channel
        for dm in range(depth_multiplier):
            for ch in range(depth_channels):
                for r in range(depth_rows):
                    o_weights += '\n' + identation
                    for c in range(depth_columns):
                        # Acceder correctamente: (depth_multiplier, channel, row, col)
                        o_weights += f'{conv_depth_weights[dm, ch, r, c]}, '
                    if comm_values:
                        original_vals = self._wrapper.depth_weights[dm, ch, r, :]
                        o_weights += f'/* {original_vals} */'

        # Remover última coma
        id_pos = o_weights.rfind(',')
        if id_pos != -1:
            o_weights = o_weights[:id_pos] + o_weights[id_pos + 1:]

        # Generar depth_bias
        d_biases = ''
        for i in range(len(conv_depth_biases)):
            d_biases += '\n' + identation + f'{conv_depth_biases[i]}, '
            if comm_values:
                d_biases += f'/* {raw_depth_biases[i]} */'

        # Remover última coma
        id_pos = d_biases.rfind(',')
        if id_pos != -1:
            d_biases = d_biases[:id_pos] + d_biases[id_pos + 1:]

        o_code = f'''
            static {data_type} depth_weights[]={{{o_weights}
            }};
            static {biases_type} depth_bias[]={{{d_biases}
            }};

            static filter_t point_filters[{point_filters}];
            '''
        init_conv_layer += o_code

        # Generar point_filters
        for i in range(point_filters):
            o_weights = ""
            for ch in range(point_channels):
                o_weights += f'{conv_point_weights[i, ch, 0, 0]}, '

            if comm_values:
                comm_weights = f' /* {self._wrapper.point_weights[i, :, 0, 0]} */'
                comm_bias = f' /* {self._wrapper.biases[i]} */'
            else:
                comm_weights = ''
                comm_bias = ''

            o_code = f'''
            static {data_type} point_weights{i}[]={{{o_weights}{comm_weights}
            }};
            static filter_t point_filter{i} = {{point_weights{i}, {conv_point_biases[i]}{comm_bias}}};
            point_filters[{i}] = point_filter{i};
            '''
            init_conv_layer += o_code

        # Generar estructura final
        init_conv_layer += f'''
            {struct_type} layer = {{{point_filters}, point_filters, {point_channels}, {point_kernel_size}, depth_weights, depth_bias, {total_depth_filters}, {depth_kernel_size}, {padding}, {strides}{qparams} }};
            return layer;
        }}'''

        return init_conv_layer


    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
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
        return f'''separable_conv2d_layer({self.name}_data, {input_name}, &{output_name});'''