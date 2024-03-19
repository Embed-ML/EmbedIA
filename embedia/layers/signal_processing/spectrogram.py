from embedia.core.layer import Layer


class Spectrogram(Layer):

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
        self.input_data_type = "data1d_t"
        self.output_data_type = "data3d_t"

        self._use_data_structure = True  # this layer require data structure initialization


        # self.output_shape = layer.shape

        # self.input_shape = (self.layer.input_length,)
        # self.output_shape = (self.layer.n_fft,self.layer.n_mels)

        # self.melspec_export = Melspec_export(layer)

        # self.struct_data_type = 'normalization_t'

        # assign properties to be used in "function_implementation"
        # self.weights = layer.get_weights()[0]
        # self.biases = layer.get_weights()[1]

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


    def calculate_memory(self, types_dict):
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
        # dt_size = ModelDataType.get_size(self.options.data_type)


        # mem_size = (n_input * dt_size/8 + sz_neuron_t) * n_neurons

        # return mem_size
        return 0

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
        text = f'''
spectrogram_layer_t init_melspec_data(void){{
    spectrogram_layer_t layer_spec;
    layer_spec.convert_to_db = {0};
    layer_spec.n_fft = {self.wrapper.n_fft};
    layer_spec.n_mels = {self.wrapper.n_mels};
    layer_spec.frame_length = {self.wrapper.input_length};
    layer_spec.sample_rate = {self.wrapper.input_fs};
    layer_spec.n_blocks = {self.wrapper.n_blocks};
    layer_spec.n_fft_table = {int(self.wrapper.n_fft / 2)};
    layer_spec.noverlap = {self.wrapper.noverlap};
    layer_spec.step = {self.wrapper.step};
    layer_spec.len_nfft_nmels = {(self.wrapper.n_fft // 2) // self.wrapper.n_mels};
    layer_spec.spec_size = {self.wrapper.shape[0] * self.wrapper.shape[1]};
    layer_spec.ts_us = {int(1 / self.wrapper.input_fs * 1000 * 1000)};

    return layer_spec;
}}
        '''
        return text

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

        return f'''create_spectrogram(melspec_data, {input_name}, &{output_name});
'''
