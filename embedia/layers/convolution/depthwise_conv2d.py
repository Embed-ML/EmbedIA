from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.core.padding_types import PaddingType
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class DepthwiseConv2D(NeuralNetLayer):
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
     rule is recomended: LayerClassName+"_layer". Example for DepthwiseConv2D class should
     be named depthwise_conv2d_layer.
     The invoke function receives an input and an output parameter with the parameter's
     name that are used in the predict function of the model.

     The DepthwiseConv2D convolutional layer is a layer that requires additional data structure
     (weights to be initialized) in addition to the input data. For this reason
     sets "_use_data_structure" to True. Because ot this, code generator generates C code
     automatically based on the content of the properties that store c code:
     - struct_data_type [automatic named]: name of data type of structure to store parameters
       like filters, kernel size, padding, etc. This structure must be declared in some .h file.
       Example: for Classname+"_layer_t" generates depthwise_conv2d_layer_t
     - variable_declaration [automatic generated]: variable declaration to store parameters.
       Example: for Classname+"_layer_t" LayerName+"_data" generates depthwise_conv2d_layer_t depthwise_conv2d_0_data
     - function_prototype [automatic generated]: function prototype to invoke on data initialization.
       Example: for struct_data_type "init_"+LayerName+"_data"(void)' generates
       depthwise_conv2d_layer_t init_conv2d_data(void)
     - variable_initialization [automatic generated]: code to initialize structure variable via
       initialization function. Example: for LayerName+"_data" = "init_"+LayerName+"_data(void)"
       generates conv2d_0_data = init_depthwise_conv2d_0_data(void).
     - function_implementation [user generated]: full code of initialization function. User must
       generate code to initialize the data structure.

     Layer wrapper required properties:
         - padding => 0=valid, 1=same
         - strides => (height, width)
         - depth_weights => 4d array formatted: filters, channel, row, column
         - point_weights => 4d array formatted: filters, channel, row, column
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
        # estimate amount multiplication and addition operations
        out_size = self.output_size

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self._wrapper.weights.shape
        MACs = out_size*n_cols*n_rows*n_channels

        #n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        #MACs += out_size*n_cols*n_rows*n_channels

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
        n_channels, n_filters, n_rows, n_cols = self._wrapper.weights.shape
        depth_params = n_channels * n_filters * n_rows * n_cols

        #n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        #point_params = n_channels * n_filters * n_rows * n_cols

        # EmbedIA filter structure size
        sz_filter_t = 4 # 'filter_t'

        # base data type in bits: float, fixed (32/16/8)
        dt_size = self.options.data_type.size
        if self.options.data_type == ModelDataType.BINARY:
            dt_size = 32

        mem_size = ((depth_params + n_filters) * dt_size / 8 +
                    sz_filter_t * n_filters)

        return mem_size

    @property
    def function_implementation(self):
        """
        Generate C code for the depthwise conv2d layer initialization function.
        Uses CBuilder.add_array and add_struct for clean, declarative generation.

        Note: depthwise conv2d has a single filter (depth_filters=1), one set of
        weights per channel, so weights[] is a flat array of all channel kernels.
        """
        weights = self._wrapper.weights
        biases = self._wrapper.biases
        depth_filters, depth_channels, depth_rows, depth_columns = weights.shape

        (weight_type, weight_converter) = self.model.get_type_converter()
        conv_weights = weight_converter.fit_transform(weights)

        if self.options.data_type == ModelDataType.QUANT8:
            (biases_type, biases_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            conv_biases = biases_converter.transform(biases)
            params = weight_converter.export_params(mode='q15')
            qparams = f', {{ {params.scale_q}, {params.zero_point} }}'
        else:
            (biases_type, biases_converter) = self.model.get_type_converter()
            conv_biases = biases_converter.transform(biases)
            qparams = ''

        strd_rows, strd_cols = self._wrapper.strides[-2], self._wrapper.strides[-1]
        assert strd_rows == strd_cols, "Only equal strides supported in row and column dimensions"

        padding = '%d' % self._wrapper.padding
        strides = f'{{{strd_rows}, {strd_cols}}}'
        kernel_size = f'{{ {depth_rows}, {depth_columns} }}'
        use_comments = self.options.data_type != ModelDataType.FLOAT
        name = self.name
        cb = self.c_builder

        cb.add()
        with cb.bgn(f'{self.struct_data_type} init_{name}_data(void)'):

            # --- weights: all channels flat, depth_columns values per row ---
            # layout: [ch0_r0, ch0_r1, ..., ch1_r0, ch1_r1, ...]
            flat_weights = [
                conv_weights[0, ch, r, c]
                for ch in range(depth_channels)
                for r in range(depth_rows)
                for c in range(depth_columns)
            ]
            row_comments = [
                str(weights[0, ch, r, 0:depth_columns])
                for ch in range(depth_channels)
                for r in range(depth_rows)
            ] if use_comments else None

            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {weight_type}',
                'weights',
                flat_weights,
                cols=depth_columns,
                comments=row_comments,
                header_comment=f'[{depth_channels} x {depth_rows} x {depth_columns}]'
            )

            cb.add()

            # --- biases: one per channel ---
            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {biases_type}',
                'biases',
                list(conv_biases),
                comments=[str(biases[ch]) for ch in range(depth_channels)] if use_comments else None,
                header_comment=f'[{depth_channels}]'
            )

            cb.add()

            # --- layer struct ---
            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {self.struct_data_type}',
                'layer',
                [
                    f'weights, biases, {depth_channels}',  # data + channels
                    f'{kernel_size}, {padding}, {strides}{qparams}'  # geometry
                ]
            )

            cb.add()
            cb.add('return layer;')

        return cb.get_code()

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA DepthwiseConv2D class associated to the Keras
        Dense layer, the function "depthwise_conv2d_layer" must be implemented in
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

        return f'''depthwise_conv2d_layer({self.name}_data, {input_name}, &{output_name});'''