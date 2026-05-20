from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.core.layer import EmbediaFile
from embedia.model_generator.project_options import ModelDataType
from embedia.core.type_converters import UnsignedFixedTypeConverter


class SpectrogramBaseLayer(NeuralNetLayer):

    """
    
    """

    def __init__(self, model, wrapper, **kwargs):
        # super().__init__(model, layer, options, **kwargs)
        # self.name = 'spectrogram'
        # self.class_name = 'spectrogram'
        # layer._class.name_ = 'spectrogram'
        # self.input_shape = self.layer.input_shape
        # self.output_shape = self.layer.output_shape
        super().__init__(model, wrapper, **kwargs)
        # self.input_data_type = "data1d_t"
        # self.output_data_type = "data3d_t"

        self._use_data_structure = True  # this layer require data structure initialization
        self._struct_data_type = 'spectrogram_layer_t'

        # self.output_shape = layer.shape

        # self.input_shape = (self.layer.input_length,)
        # self.output_shape = (self.layer.n_fft,self.layer.n_mels)

        # self.melspec_export = Melspec_export(layer)

        # assign properties to be used in "function_implementation"
        # self.weights = layer.get_weights()[0]
        # self.biases = layer.get_weights()[1]

    @property
    def required_files(self):
        '''
        retorna una lista de tuplas indicando los nombres de los archivos donde se encuentra la definicion de
        tipos de datos (.h) y la implementación de las funciones (.c) requeridos por la capa/elemento
        '''
        return super().required_files + [(EmbediaFile('signals.h'), EmbediaFile('signals.c'))]


    def get_input_shape(self):
        """
        Returns the shape of the input data. This method is redefined because
        SKLearn "Scalers" do not have the "input_shape" property of the Keras
        layers on which the original implementation is based.

        Returns
        -------
        n-tuple
            shape of the input data
        """
        
        return self.wrapper.input_shape

    def get_output_shape(self):
        """
        Returns the shape of the output data.

        Returns
        -------
        n-tuple
            shape of the output data
        """
        return self.wrapper.output_shape

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # layer dimensions
        # (n_input, n_neurons) = self.weights.shape

        # MACs = n_input * n_neurons

        # MACs = self.get_input_shape()[0]
        return 0


    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        # (n_input, n_neurons) = self.weights.shape

        # # neuron structure size
        # sz_neuron_t = types_dict['neuron_t']

        # # base data type in bits: float, fixed (32/16/8)
        # dt_size = self.options.data_type.size

        # mem_size = (n_input * dt_size/8 + sz_neuron_t) * n_neurons

        # return mem_size
        return 0
    @property
    def internal_alloc_required(self) -> int:
        if len(self.output_shape) == 3:
            ch_out, h_out, w_out = self.output_shape
        else:
            ch_out = 1
            h_out, w_out = self.output_shape

        if self.options.data_type == ModelDataType.QUANT8:
            data_size = 16
        else:
            data_size = self.options.data_type.size

        size_out = ch_out * h_out * w_out * (data_size // 8)

        # data_re y data_im son siempre int32_t independiente del tipo fixed
        # porque la FFT opera internamente con mayor precisión. Para float tambien es 4 bytes
        size_fft_buffers = 2 * self.wrapper.frame_length * 4  # sizeof(int32_t) = 4

        return size_out + size_fft_buffers

    @property
    def function_implementation(self):
        """
        Generate C code with the initialization function of the additional
        structure (defined in "neural_net.h") required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """
#         text = f'''// File spectrogram.h
# #ifndef _SPECTROGRAM_H
# #define _SPECTROGRAM_H

# #include "fft.h"

# // Constantes autogeneradas
# #ifndef N_FFT
# #define CONVERT_TO_DB {0}
# #define N_FFT {self.layer.n_fft}
# #define N_MELS {self.layer.n_mels}
# #define FRAME_LENGTH {self.layer.input_length}
# #define SAMPLE_RATE {self.layer.input_fs}
# #define BLOCKS {self.layer.n_blocks}
# #define N_FFT_TABLE {int(self.layer.n_fft/2)}
# #define NOVERLAP {self.layer.noverlap}
# #define STEP {self.layer.step}
# #define LEN_NFFT_NMELS {(self.layer.n_fft//2)//self.layer.n_mels}
# #define SPEC_SIZE {self.layer.shape[0]*self.layer.shape[1]}
# #define TS_US {int(1/self.layer.input_fs*1000*1000)}
# #endif

# void create_spectrogram(float *data, float *result);

# #endif
#     '''
        cb = self.c_builder

        top_db_limit = self.wrapper.top_db
        win_data = self.wrapper.window[:self.wrapper.window.shape[0] // 2]
        win_type = 'window_t'
        if self.options.data_type in [ModelDataType.FIXED8, ModelDataType.FIXED16, ModelDataType.QUANT8]:
            # Special converter for Q0.8 data format, for fixed point data types
            win_converter = UnsignedFixedTypeConverter(0, 8)
            win_shift = 8
        elif self.options.data_type == ModelDataType.FIXED32:
            win_converter = UnsignedFixedTypeConverter(0, 16)
            win_shift = 16
        else:
            (_, win_converter) = self.model.get_type_converter()
            win_shift = 0

        win_converter.fit(win_data)
        cvt_window = win_converter.transform(win_data)
        window = cb.to_array(cvt_window)

        if self.wrapper.convert_to_db:
            if self.options.data_type == ModelDataType.QUANT8:
                (_,db_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            else:
                (_,db_converter) = self.model.get_type_converter()
            top_db = db_converter.fit_transform(top_db_limit)
        else:
            top_db = 0


        text = f'''
spectrogram_layer_t init_stft_data(void){{
    static const {win_type} window[] = {{{window}}};

    spectrogram_layer_t layer_spec;
    layer_spec.n_channels = {self.wrapper.n_channels};
    layer_spec.sample_rate = {self.wrapper.sample_rate};
    layer_spec.ts_us = {int(1 / self.wrapper.sample_rate * 1000 * 1000)};
    layer_spec.frame_length = {self.wrapper.frame_length};
    layer_spec.overlap_length = {self.wrapper.overlap_length};
    layer_spec.hop_length = {self.wrapper.hop_length};
    layer_spec.window = window;
    layer_spec.window_shift = {win_shift};
    layer_spec.n_fft_table = {int(self.wrapper.frame_length / 2)};    
    layer_spec.n_frames= {self.wrapper.n_frames};
    layer_spec.spec_size = {self.wrapper.shape[0] * self.wrapper.shape[1]};
    layer_spec.convert_to_db = {1 if self.wrapper.convert_to_db else 0};
    layer_spec.top_db = {top_db};

    return layer_spec;
}}
        '''
        # layer_spec.len_nfft_nmels = {(self.wrapper.n_fft // 2) // self.wrapper.n_mels};
        return text

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
        fn_name = self.name
        if len(self.input_shape) > 1:
            fn_name = 'multi_'+fn_name
        return f'''{fn_name}_layer(stft_data, {input_name}, &{output_name});
'''
