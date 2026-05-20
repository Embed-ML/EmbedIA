from embedia.core.neural_net_layer import NeuralNetLayer


class Pooling(NeuralNetLayer):
    """
    The Pooling layer is a layer that requires additional data beyond the input
    data. However, these values can be assigned directly to the additional data
    structure in its declaration. For this reason, it inherits from the "Layer"
    class that implements the basic behavior of an EmbedIA layer/element and
    not from the "DataLayer" class that implements an initialization function.
    This structure is declared as static in the before the invocation of the C
    function that implements the layer function.
    Normally, the programmer must implement the "predict" method, with the
    invocation to the EmbedIA function (previously implemented in "neural_net.c")
    that performs the layer processing.
    This class encompasses the behavior of the pooling layers (Average, Max,
    etc.) and in principle it is not necessary to create subclasses for the
    implementation of each type. In particular it implements the
    "get_pool_name" method that uses an automatic naming rule from the name of
    the Keras pooling function for all pooling layers. The "predict" method
    invokes the C function (which must be defined in "neural_net.h" and
    implemented in "neural_net.c") using this name.
    Ex: For the pooling function named "avg_pool" for 2 dimensions, the C
    function named "avg_pooling2d_layer" will be called, composed by the first
    part of the name "avg" followed by "_pooling" + "input dimension" +
    "d_layer".

    Layer wrapper required properties:
        - strides => 2D=(height, width), 1D=length
        - pool_size =>  2D=(height, width), 1D=length
        - dimensions => the pooling dimensions 1D=1, 2D=2, 3D=3
        - function_name => function name for pooling layers ex: "avg_pooling2d_layer"
    """

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

    def calculate_ACOPS(self):
        """
        ACOPs => Arithmetic and Comparison Operations
        Calculates ACOPS (non-MACC) operations for Pooling layers (1D, 2D, 3D),
        including Max, Average, and Global pooling.

        Accounts for pool size, stride, and padding to compute:
          - Comparison operations (MaxPool): (pool_volume - 1) per output element
          - Arithmetic operations (AvgPool): pool_volume additions + 1 division per output
          - Memory operations: input reads (pool_volume per output) + output writes

        Returns
        -------
        int
            Total ACOPS operations (pooling_ops + memory_ops)
        """

        pool_type = self._wrapper.function_name.lower()
        total_output_elements = self.output_size  # already precomputed in wrapper

        # --- 1. Determinar pool_volume ---
        if self._wrapper.is_global:
            # Global pooling cubre toda la dimensión espacial
            input_shape = self.input_shape[1:]  # excluir batch
            dims = self._wrapper.dimensions
            # input_shape = (W,), (H, W, C), (D, H, W, C), etc.
            if dims == 1:
                pool_volume = input_shape[0]
            elif dims == 2:
                pool_volume = input_shape[0] * input_shape[1]
            elif dims == 3:
                pool_volume = input_shape[0] * input_shape[1] * input_shape[2]
            else:
                raise ValueError("Unsupported pooling dimensions for global pooling")
        else:
            # Pool normal: usar pool_size
            pool_size = self._wrapper.pool_size
            pool_volume = 1
            for k in pool_size:
                pool_volume *= k

        # --- 2. Pooling-specific operations ---
        if 'max' in pool_type:
            pooling_ops = total_output_elements * (pool_volume - 1)
        elif 'avg' in pool_type or 'average' in pool_type:
            pooling_ops = total_output_elements * (pool_volume + 1)  # sumas + división
        else:
            pooling_ops = 0  # fallback

        # --- 3. Memory operations ---
        total_input_reads = total_output_elements * pool_volume
        total_output_writes = total_output_elements
        memory_ops = total_input_reads + total_output_writes

        return pooling_ops + memory_ops


    @property
    def pool_name(self):
        """
        Gets the name of the EmbedIA function to be invoked to perform the
        layer processing. The definition of the function with this name must
        be defined in some ".h" and implemented in respective ".c".

        Parameters
        ----------
        layer : object
            pooling layer object

        Returns
        -------
        str
            name of EmbedIA pooling function to call in predict method

        """
        return '%s_pooling%dd_layer' % (self._wrapper.function_name, self._wrapper.dimensions)


    @property
    def layer_type_name(self):
        return f'{self.__class__.__name__}({self.wrapper.function_name})'

    @property
    def struct_data_type(self):
        """
        gets automatic embedia name for structure associated with layer/element
        Returns
        -------
        str
            embedia type name for layer/element.
        """
        return 'pooling%dd_layer_t' % self._wrapper.dimensions

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invoke the EmbedIA function that implements
        the layer/element. The C function must be previously implemented in
        "neural_net.c" and by convention should be called by the name
        autogenerated "get_pool_name" method.

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

        name = self.name
        pool_name = self.pool_name
        dim = self._wrapper.dimensions
        if not self._wrapper.is_global:
            strides = self._wrapper.strides[0]
            pool_size = self._wrapper.pool_size[0]
            text = f'''static const pooling{dim}d_layer_t {name}_data = {{ {pool_size}, {strides} }};
{pool_name}({name}_data, {input_name}, &{output_name});'''
        else:
            text = f'''{pool_name}({input_name}, &{output_name});'''

        return text
