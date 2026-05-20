
from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.core.padding_types import PaddingType
from embedia.model_generator.project_options import ModelDataType

import numpy as np


class Conv1D(NeuralNetLayer):
    """
    1D Convolutional layer implementation for EmbedIA.

    This layer performs 1D convolution operations commonly used for:
    - Time series analysis
    - Audio signal processing
    - NLP sequence processing
    - Sensor data analysis

    The Conv1D layer requires additional data structure (weights/biases) and
    sets "_use_data_structure" to True for automatic C code generation.

    Layer wrapper required properties:
        - padding => 0=valid, 1=same, 2=causal
        - strides => stride value (integer)
        - weights => 3d array formatted: filters, channels, kernel_size
        - biases => 1d array
        - kernel_size => size of convolution kernel
        - dilation_rate => dilation factor
    """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)
        self._use_data_structure = True  # this layer requires data structure initialization

    def calculate_params(self):
        """
        Calculates the number of trainable and non-trainable parameters in a 1D convolutional layer.

        Returns
        -------
        tuple
            (number_of_trainable_params, number_of_non_trainable_params)
        """
        # Shape convention: (num_filters, num_channels, kernel_size)
        n_filters, n_channels, kernel_size = self.wrapper.weights.shape

        # Kernel parameters: each filter has (n_channels × kernel_size) weights
        trainable_kernels = n_filters * n_channels * kernel_size
        trainable_biases = n_filters  # One bias per output filter

        total_trainable = trainable_kernels + trainable_biases
        total_non_trainable = 0  # Standard Conv1D layers have no non-trainable parameters

        return (total_trainable, total_non_trainable)

    def calculate_MAC(self):
        """
        Calculates the number of Multiply-Accumulate (MAC) operations in a 1D convolutional layer.

        For a conv1d layer, each output element requires:
        - n_channels * kernel_size multiplications
        - (n_channels * kernel_size - 1) additions
        (Each multiply-add pair counts as 1 MAC)

        Returns
        -------
        int
            Total number of MAC operations for the layer
        """
        # Shape: (num_filters, num_channels, kernel_size)
        n_filters, n_channels, kernel_size = self._wrapper.weights.shape

        # Calculate total MAC operations:
        # = output_length * n_filters * n_channels * kernel_size
        total_MACs = self.output_size * n_channels * kernel_size

        return total_MACs

    def calculate_ACOPS(self):
        """
        Calculates the number of non-MACC operations (ACOPS) in a Conv1D layer:
        - Bias additions (arithmetic)
        - Memory access operations (load/store)

        Returns
        -------
        int
            Total count of non-MACC operations (ACOPS)
        """
        # Shape: (num_filters, num_channels, kernel_size)
        n_filters, _, _ = self.wrapper.weights.shape

        output_size = self.output_size  # Get output length * filters

        # Bias additions: 1 addition per output element (if bias exists)
        bias_ops = output_size if len(self.wrapper.biases) > 0 else 0

        # Memory operations: minimum 2 memory ops per output element
        memory_ops = 2 * output_size

        return bias_ops + memory_ops

    def calculate_memory(self):
        """
        Calculates amount of memory required to store the data of layer.

        Returns
        -------
        int
            Amount of memory required in bytes
        """
        # Layer dimensions
        n_filters, n_channels, kernel_size = self._wrapper.weights.shape

        # EmbedIA filter structure size - estimated
        sz_filter_t = 8  # Pointer + bias (4+4 bytes)

        # Base data type in bits: float, fixed (32/16/8)
        dt_size = self.options.data_type.size

        mem_size = (n_channels * kernel_size * dt_size / 8 + sz_filter_t) * n_filters

        return mem_size

    @property
    def function_implementation(self):
        """
        Generate C code for the conv1d layer initialization function.
        Uses CBuilder.add_array and add_struct for clean, declarative generation.
        """
        weights = self._wrapper.weights
        biases = self._wrapper.biases
        padding = f'{self._wrapper.padding}'
        stride = f'{self._wrapper.strides}'
        kernel_size = f'{self._wrapper.kernel_size}'
        dilation_rate = f'{self._wrapper.dilation_rate}'

        n_filters, n_channels, k_size = weights.shape

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

        use_comments = self.options.data_type != ModelDataType.FLOAT
        name = self.name + '_data'
        cb = self.c_builder

        cb.add()
        with cb.bgn(f'{self.struct_data_type} init_{name}(void)'):

            # --- weight arrays: one per filter, laid out as [channels x k_size] ---
            for i in range(n_filters):
                flat_weights = conv_weights[i].flatten().tolist() \
                    if hasattr(conv_weights[i], 'tolist') \
                    else list(conv_weights[i].flatten())

                row_comments = None
                if use_comments:
                    row_comments = [
                        f'ch{ch} | {list(weights[i, ch, :])}'
                        for ch in range(n_channels)
                    ]

                bias_comment = [str(biases[i])] if use_comments else None

                cb.add_array(
                    f'static EMBEDIA_MODEL_STORAGE {weight_type}',
                    f'weights{i}',
                    flat_weights,
                    cols=k_size,
                    comments=row_comments,
                    header_comment=f'[{n_channels} x {k_size}]'
                )

            cb.add()

            # --- filters array: weights pointer + bias per filter ---
            cb.add_array(
                'static EMBEDIA_MODEL_STORAGE filter_t',
                'filters',
                [f'{{ weights{i}, {conv_biases[i]} }}' for i in range(n_filters)],
                comments=[str(biases[i]) for i in range(n_filters)] if use_comments else None
            )

            cb.add()

            # --- layer struct ---
            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {self.struct_data_type}',
                'layer',
                [
                    f'{n_filters}, filters, {n_channels}',  # n_filters, filters, channels
                    f'{kernel_size}, {padding}, {stride}{qparams}'  # kernel, padding, stride
                ]
            )

            cb.add()
            cb.add('return layer;')

        return cb.get_code()

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element.

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
        # Choose optimized function name based on layer parameters
        if self._wrapper.padding == PaddingType.SAME:
            opt_name = '_padding'
        elif self._wrapper.strides>1:
            opt_name = '_strides'
        else:
            opt_name = ''
        return f'''conv1d{opt_name}_layer({self.name}_data, {input_name}, &{output_name});'''