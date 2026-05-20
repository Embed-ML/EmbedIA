from embedia.core.neural_net_layer import NeuralNetLayer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType
import numpy as np


class Dense(NeuralNetLayer):
    """
    The Dense layer is a layer that requires additional data (weights to be
    initialized) in addition to the input data. For this reason it inherits
    from DataLayer which generates C code automatically with the variable that
    will store the data, the declaration of the prototype of the initialization
    function and the call to it.
    Normally the programmer must implement two methods. The first one is
    "function_implementation" which returns the implementation of the initialization
    function in C code, retrieving the layer information and dumping it into
    the structure (defined in neural_net.h") in an appropriate way. The second one
    is "predict" where the programmer must invoke the EmbedIA function
    (implemented in "neural_net.c") that must perform the processing of the layer.
    To avoid overlapping names, both the function name and the variable name
    are generated automatically using the layer name. The same happens with the
    data type of the structure to be completed whose name comes from the name
    of the Python class that implements the layer.
    Ex: As this class is called Dense, the type of the additional structure
    will be called "dense_datat" and must be defined previously in the
    "neural_net.h" file.
    If the name of the layer is dense0, it will automatically be generated in
    the C file of the model, the declaration of the variable
    "dense_datat dense0_data", the prototype of the initialization function
    "dense_datat init_dense0_data(void)" and the invocation
    "dense0_data = init_dense0_data()". This way of naming must be taken into
    account in the implementation of the initialization function in the
    "function_implementation" method
    """
    support_quantization = False  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization


    def calculate_params(self):
        """
         calculates trainable and non trainable parameters of layer
         Returns
         -------
         int
             tuple (#trainable params, # non trainable params)

         """

        trainable = self.wrapper.weights.size + self.wrapper.biases.size
        non_trainable = 0

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        # layer dimensions
        (n_input, n_neurons) = self._wrapper.weights.shape

        MACs = n_input * n_neurons

        return MACs

    def calculate_ACOPS(self):
        """
        Calculates the number of non-MACC operations (ACOPS) in a Dense layer:
        - Bias additions (arithmetic).
        - Memory load/store operations (if applicable).

        Returns
        -------
        int
            Total non-MACC operations (ACOPS).
        """
        # Layer dimensions
        n_neurons = self._wrapper.weights.shape[1]

        # - Bias additions   : 1 add by output neuron
        # - Memory operations: 2 access by neuron (1 read + 1 write)

        return (1+2) * n_neurons

    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        (n_input, n_neurons) = self._wrapper.weights.shape

        # neuron structure size
        # struct{ float * weights; float bias; }neuron_t;
        sz_neuron_t = 4

        # base data type in bits: float, fixed (32/16/8)
        dt_size = self.options.data_type.size


        mem_size = (n_input * dt_size/8 + sz_neuron_t) * n_neurons

        return mem_size

#     @property
#     def function_implementation(self):
#         """
#         Generate C code with the initialization function of the additional
#         structure (defined in "neural_net.h") required by the layer.
#         Note: it is important to note the automatically generated function
#         prototype (defined in the DataLayer class).
#
#         Returns
#         -------
#         str
#             C function for data initialization
#         """
#         weights = self._wrapper.weights
#         biases = self._wrapper.biases
#         name = self.name
#         struct_type = self.struct_data_type
#         (data_type, data_converter) = self.model.get_type_converter()
#
#         (n_input, n_neurons) = weights.shape
#
#         init_dense_layer = f'''
# {struct_type} init_{name}_data(void){{
#
#     static neuron_t neurons[{n_neurons}];
# '''
#         o_code = ''
#
#         for neuron_id in range(n_neurons):
#
#             all_weights = np.concatenate([weights[:, neuron_id], [biases[neuron_id]]])
#             (conv_weights, quant_params) = self.convert_to_embedia_data( data_converter, all_weights )
#
#             o_weights = declare_array(f'static const {data_type}', f'weights{neuron_id}', None, conv_weights[:-1])
#
#             o_code += f'''
#     /* {weights[:, neuron_id]} {biases[neuron_id]}*/
#     {o_weights};
#
#     static const neuron_t neuron{neuron_id} = {{weights{neuron_id}, {conv_weights[-1]} {quant_params} }};
#     neurons[{neuron_id}]=neuron{neuron_id};
# '''
#         init_dense_layer += o_code
#
#         init_dense_layer += f'''
#     dense_layer_t layer= {{ {n_neurons}, neurons}};
#     return layer;
# }}
# '''
#         return init_dense_layer

    @property
    def function_implementation(self):
        """
        Generate C code for the dense layer initialization function.
        Uses CBuilder.add_array and add_struct for clean, declarative generation.

        Generated C structure:
            static EMBEDIA_MODEL_STORAGE {weight_type} weights[] = {  // [n_neurons x n_input]
                ...  // one neuron per row, float values as comments if quantized
            };
            static EMBEDIA_MODEL_STORAGE {biases_type} biases[] = {  // [n_neurons]
                ...  // float values as comment if quantized
            };
            static EMBEDIA_MODEL_STORAGE dense_layer_t layer = {
                n_input, n_neurons,
                weights, biases
            };
            return layer;
        """
        cb = self.c_builder
        weights = self._wrapper.weights
        biases = self._wrapper.biases.reshape(1, -1)
        name = self.name
        n_input, n_neurons = weights.shape

        (weight_type, weight_converter) = self.model.get_type_converter()
        (conv_weights, quant_params) = self.convert_to_embedia_data(weight_converter, weights)

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

        cb.add()
        with cb.bgn(f'{self.struct_data_type} init_{name}_data(void)'):

            cb.add(f'// {n_input} inputs, {n_neurons} neurons')
            cb.add()

            # --- weights: column-major layout, one neuron per row ---
            flat_weights = []
            row_comments = [] if use_comments else None
            for i in range(n_neurons):
                flat_weights.extend(conv_weights[:, i].tolist())
                if use_comments:
                    row_comments.append(
                        f'N{i:<3d} | {cb.to_array(weights[:, i], fmt=".6f")}'
                    )

            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {weight_type}',
                'weights',
                flat_weights,
                cols=n_input,
                comments=row_comments,
                header_comment=f'[{n_neurons} x {n_input}]'
            )

            cb.add()

            # --- biases ---
            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {biases_type}',
                'biases',
                conv_biases[0].tolist(),
                comments=[cb.to_array(biases[0], fmt='.6f')] if use_comments else None,
                header_comment=f'[{n_neurons}]'
            )

            cb.add()

            # --- layer struct ---
            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {self.struct_data_type}',
                'layer',
                [
                    f'{n_input}, {n_neurons}',  # input features, neurons
                    f'weights, biases{qparams}'  # data pointers + optional qparams
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

        return f'''dense_layer(&{self.name}_data, &{input_name}, &{output_name});
'''
