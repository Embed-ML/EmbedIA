from math import sqrt
from embedia.core.layer import Layer
from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType


class BatchNormalization(Layer):
    """
    The normalization layer is a layer that requires additional data
    (coefficients and averages values to be initialized) in addition to the
    input data. For this reason it inherits from DataLayer which generates C
    code automatically with the variable that will store the data, the
    declaration of the prototype of the initialization function and the call
    to it.
    Normally the programmer must implement two methods. The first one is
    "function_implementation" which returns the implementation of the initialization
    function in C code, retrieving the layer information and dumping it into
    the structure (defined in embedia.h) appropriately. The second one is
    "invoke", where the programmer must invoke the function EmbedIA function
    (implemented in "embedia.c") that should perform the layer processing.
    To avoid overlapping names, both the function name and the variable name
    are automatically generated using the layer name. The same is true for the
    data type of the structure to be completed whose name comes from the name
    of the Python class that implements the layer.
    Ex: Since this class is called Normalization, the type of the additional
    structure must be called "normalization_datat" and must be previously
    defined in the "embedia.h" file.
    If the name of the layer is normalization0, it will be automatically
    generated in the model's C file, the declaration of the variable
    "normalization_datat normalization0_data", the prototype of the
    initialization function "normalization_datat init_normalization0_data(void)"
    and the invocation "normalization0_data = init_normalization0_data()".
    This way of naming must be taken into account in the implementation of the
    initialization function in the "function_implementation" method.
    This class implements the operation of normalizations that use averages and
    coefficients ([average-value]/coefficient ). The classes that inherit from
    this one must only fill the values of the "sub_values" and "div_values"
    properties in their constructor.

    Layer wrapper required properties:
        - gamma
        - beta
        - moving_mean
        - moving_variance
        - epsilon
"""

    # Constructor receives batch normalization object in layer
    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._inplace_output = True # EmbedIA function saves output into input
        self._use_data_structure = True  # this layer require data structure initialization

    @property
    def struct_data_type(self):
        return 'batch_normalization_layer_t'
    
    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        # batch norm has 4 data array: beta, gamma, moving mean and moving
        # variance
        n_features = len(self._wrapper.gamma)
        n_arrays = 4 - 2  # the four arrays are optimized into two (see below)

        # neuron structure size
        # struct {uint32_t length;  float *moving_inv_std_dev;  float *std_beta;} batch_normalization_layer_t;
        sz_batch_norm_t = 12

        # base data type: float, fixed, binary (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)

        mem_size = n_arrays * n_features * dt_size/8 + sz_batch_norm_t

        return mem_size

    @property
    def function_implementation(self):

        (data_type, data_converter) = self.model.get_type_converter()

        macro_converter = lambda x:x

        name = self.name
        struct_type = self.struct_data_type
        inv_gamma_dev_name = 'inv_gamma_dev'
        std_beta_name = 'std_beta'

        gamma = self._wrapper.gamma
        beta = self._wrapper.beta
        moving_mean = self._wrapper.moving_mean
        moving_variance = self._wrapper.moving_variance
        epsilon = self._wrapper.epsilon
        length = len(moving_mean)

        # Params: data type, var name, macro, array/list of values
        array_type = f'static const {data_type}'

        # gamma, beta and mov_mean can be eliminated due to the optimization performed below
        
        # Optimization to avoid a multiplication, a division and square root
        # calculation in the microcontroller
        # epsilon is a small value to avoid division by zero
        
        #gamma_variance = np.array([(gamma[i] / sqrt(moving_variance[i] + epsilon)) for i in range(gamma.size)])
        #inv_gamma_dev = ['%f/sqrt(%f+%f)' % (self.gamma[i], self.moving_variance[i], self.epsilon) for i in range(self.gamma.size)]
        inv_gamma_dev = [gamma[i] / sqrt(moving_variance[i]+epsilon) for i in range(gamma.size)]
        inv_gamma_dev = data_converter.fit_transform(inv_gamma_dev)
        qparam = f', {{ {data_converter.scale}, {data_converter.zero_pt} }}' if self.is_data_quantized else ''

        # standard_beta = np.array([(beta[i] - moving_mean[i] * standard_gamma[i]) for i in range(beta.size)])

        #std_beta = ['%f-(%f*%f/sqrt(%f+%f))' % (self.beta[i], self.moving_mean[i], self.gamma[i], self.moving_variance[i], self.epsilon) for i in range(self.beta.size)]
        std_beta = [beta[i] - (moving_mean[i]*gamma[i]/sqrt(moving_variance[i]+epsilon) ) for i in range(beta.size)]
        std_beta = data_converter.fit_transform(std_beta)
        qparam += f', {{ {data_converter.scale}, {data_converter.zero_pt} }}' if self.is_data_quantized else ''

        # get inverse of standard dev (square root of moving variance)
        o_inv_mov_std = declare_array(array_type, inv_gamma_dev_name, macro_converter, inv_gamma_dev)
        
        o_std_beta = declare_array(array_type, std_beta_name, macro_converter, std_beta)

        # By exporting this two new parameters, the layer only needs to perform a multiplication and a sum

        init_layer = f'''
{struct_type} init_{name}_data(void){{

    {o_inv_mov_std};
    {o_std_beta};

    static const {struct_type} norm = {{ {length}, {inv_gamma_dev_name}, {std_beta_name} {qparam} }};
    return norm;
}}
'''
        return init_layer

    def invoke(self, input_name, output_name):
        dim = len(self.input_shape)
        return f'''batch_normalization{dim}d_layer({self.name}_data, &{output_name});'''
