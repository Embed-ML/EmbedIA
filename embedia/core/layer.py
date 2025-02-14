import regex as re
import numpy as np
import tensorflow.keras.backend as K

class LayerInfo(object):
    """
    This class defines a container for layer information:
        name, type name, activation function name, #params, MACs, output shape
    """

    def __init__(self, embedia_layer, types_dict):
        self.set_layer(embedia_layer, types_dict)

    def set_layer(self, layer, types_dict):
        self.layer = layer
        self.types_dict = types_dict

        self._update_properties()

    def _update_properties(self):
        embedia_layer = self.layer

        self.class_name = embedia_layer.__class__.__name__
        self.layer_name = embedia_layer.name

        self.output_shape = self.layer.output_shape

        k_layer = self.layer.wrapper
        if hasattr(k_layer, 'trainable_weights'):
            trainable = int(np.sum([K.count_params(p) for p in k_layer.trainable_weights]))
            non_trainable = int(np.sum([K.count_params(p) for p in k_layer.non_trainable_weights]))
        else:
            trainable = 0
            non_trainable = 0

        self.params = (trainable, non_trainable)

        self.macs_ops = self.layer.calculate_MAC()

        self.memory = self.layer.calculate_memory()


class Layer(object):
    """
    Develop info:
    This class defines the basic behavior of an EmdedIA layer/element. It defines
    methods/properties to obtain information related to the inputs and outputs of
    the layer/element such as its shape, number of elements, EmbedIA associated
    data type.
    It also implements methods to generate the C code necessary for debugging
    function and invocation of the C function associated to the layer/element.
    This function must be implemented in some .c file its prototype declared
    in respective .h. The name of function can be anything but an EmbedIA naming
    rule is recomended: LayerClassName+"_layer". Example for Conv2D class should
    be named conv2d_layer.
    The invoke function receives an input and an output parameter with the parameter's
    name that are used in the predict function of the model.
      - model = None # EmbedIA model
      - target = None # object for data access
      - support_quantization = False # support quantized data. Default False
      - inplace_output = False # process output in input variable
      - use_data_structure = False # requires data structure definition an initialization
    """

    def __init__(self, model, wrapper, **kwargs):
        """
        Constructor that receives:
            - EmbedIA Model
            - object (Keras, SkLearn, etc.) associated to the EmbedIA layer
        Parameters
        ----------
        wrapper : object
            layer/object is associated to this EmbedIA layer/element. For
            example, it can receive a Keras layer or a SkLearn scaler.
        Returns
        -------
        None.
        """
        super().__init__(**kwargs)
        self._model = model
        self._wrapper = wrapper

        # assign layer name from layer/element associated, if it's possible
        self._name = model.get_unique_name(self)

        self._input_shape = None
        self._output_shape = None

        self._use_data_structure = False
        # When the value of this property is "true" it indicates that the
        # layer/element can process the output result on the same input
        # parameter. A typical case are layers that perform normalization
        # or activation functions.
        self._inplace_output = False

    @property
    def name(self):
        return self._name

    @property
    def options(self):
        return self._model.options

    @property
    def model(self):
        return self._model

    @property
    def wrapper(self):
        return self._wrapper

    @property
    def inplace_output(self):
        return self._inplace_output

    @property
    def support_quantization(self):
        return self._support_quantization


    @property
    def use_data_structure(self):
        return self._use_data_structure

    @property
    def struct_data_type(self):
        """
        gets automatic embedia name for structure associated with layer/element
        Returns
        -------
        str
            embedia type name for layer/element.
        """
        return self.embedia_type_name + '_layer_t'


    @property
    def variable_declaration(self):
        """
        Generates C code with the declaration of a variable of the type
        indicated by the property "struct_data_type" and with the name
        "layer name "+"_data" (e.g.: "dense_data_t dense0_data;").
        Parameters
        ----------
            None
        Returns
        -------
        str
            C code declaring a variable that stores the additional data
            required (e.g. neuron weights) to perform the function of the
            layer/element
        """

        return f'{self.struct_data_type} {self.name}_data;\n'

    @property
    def function_prototype(self):
        """
        Generates C code with the declaration of the prototype of the function
        that must initialize the data structure required by the layer
        (e.g. neuron weights). The return value must match the type
        indicated by the property "struct_data_type" and function name must
        have the name "init_"+"layer name "+"_data" and defined in "neural_net.c"
        file (e.g.: "dense_data_t init_dense0_data();").
        Returns
        -------
        str
            C code with the function prototype for data initialization
        """

        return f'{self.struct_data_type} init_{self.name}_data(void);\n'

    @property
    def variable_initialization(self):
        """
        Generates C code with the implementation of the function that must
        initialize the data structure required by the layer (e.g. neuron
        weights). The return value must match the type indicated by the
        property "struct_data_type". The function must be named
        "init_"+"layer name "+"_data" and must be implemented in "neural_net.c"
        file (e.g.: "dense_data_t init_dense0_data() {....}").
        Note that data types such as arrays must be declared as "static" to
        persist in memory after the function has been invoked.
        Returns
        -------
        str
            DESCRIPTION.
        """
        return f'    {self.name}_data = init_{self.name}_data();\n'

    @property
    def function_implementation(self):
        """
        this python method must implement the C function defined as prototype
        in "prototypes_init" in order to return the structure defined in
        "struct_data_type" property filled with the layer data returns a string
        with the C implementation of the function declared in "function_prototype"m
        ethod. This function must return a filled structure of type defined in
        "struct_data_type" property with de layer data.
        It should be noted that array type data must be declared as static.
        For mor details se implementation of Dense layer.
        Parameters
        ----------
            None
        Returns
        -------
        str
            C code with the implementation of the function that fills the
            structure required by the layer/element embedia.
        """
        return ''


    def get_class_name(self):
        """
        get the name of this class. Usefull for automatic name generation
        Returns
        -------
        str
            name of class
        """
        return type(self).__name__

    @property
    def embedia_type_name(self):
        """
        generates an automatic EmbedIA name for data type assosiated to the
        class that implements layer/element function. This name must exits in
        "neural_net.h" and has snake case format
        Returns
        -------
        str
            Embedia type name for layer/element.
        """
        name = self.get_class_name()
        # name in snake case
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        # snake case but skip it in 1D, 2D, etc. to avoid 1_d, 2_d
        name = re.sub(r'(\d)_d', r'\1d', name)

        return name

    @property
    def is_data_quantized(self):
        return self.model.is_data_quantized

    @property
    def is_quantizable(self):
        return self._support_quantization

    def convert_to_embedia_data(self, data_converter, values, fit=True):
        if fit:
            data_converter.fit(values)
        conv_values = data_converter.transform(values)
        if self.is_data_quantized:
            quant_params = f', {{ {data_converter.scale}, {data_converter.zero_pt} }}'
        else:
            quant_params = ''
        return (conv_values, quant_params)


    @property
    def input_data_type(self):
        """
        returns the C data type used as input by the EmbedIA layer/element.
        Returns
        -------
        str
            type name of the input for layer/element
        """
        return 'data%dd_t' % len(self.input_shape)

    @property
    def output_data_type(self):
        """
        get the C data type used as output by the EmbedIA layer/element.
        Returns
        -------
        str
            type name of the output for layer/element
        """
        return 'data%dd_t' % len(self.output_shape)

    @property
    def input_shape(self):
        """
        Get the shape of the input to the EmbedIA layer/element.

        If the value of _input_shape is None (the default set in the constructor),
        it delegates to get the value from the wrapper (if a wrapper is set).

        If the Layer itself cannot define the shape, nor can the wrapper, then the
        EmbediaModel must set _input_shape in order to get the shape (necessary to
        determine the input data type of the C++ EmbedIA layer prototype) after all
        layers of model has been created.

        Returns
        -------
        n-tuple
            returns the input shape of the layer/element.
        """
        if self._input_shape is None:
            if self._wrapper is None:
                return None
            else:
                shape = self._wrapper.input_shape
        else:
            shape = self._input_shape

        if len(shape) >= 1 and shape[0] is None:
            return shape[1:]
        return shape

    @input_shape.setter
    def input_shape(self, value):
        """
        Set the input shape of the EmbedIA layer/element.

        Sets the _input_shape attribute to the provided value.
        If _input_shape is None (the default set in the constructor), the getter method will attempt
        to get the value from the wrapper's input_shape property (if a wrapper is set). Otherwise,
        the getter will return the _input_shape value.

        Parameters
        ----------
        value : n-tuple
            The new input shape value to set.

        """
        self._input_shape = value


    @property
    def output_shape(self):
        """
        If the value of _output_shape is None (the default set in the constructor),
        it delegates to get the value from the wrapper (if a wrapper is set).

        If the Layer itself cannot define the shape, nor can the wrapper, then the
        EmbediaModel must set _output_shape in order to get the shape (necessary to
        determine the output data type of the C++ EmbedIA layer prototype) after all
        layers of model has been created.

        Returns
        -------
        n-tuple
            returns the input shape of the layer/element.
        """

        if self._output_shape is None:
            if self._wrapper is None:
                return None
            else:
                shape = self._wrapper.output_shape
        else:
            shape = self._output_shape

        if len(shape) >= 1 and shape[0] is None:
            return shape[1:]
        return shape

    @output_shape.setter
    def output_shape(self, value):
        """
         Set the input shape of the EmbedIA layer/element.

         Sets the _output_shape attribute to the provided value.
         If _output_shape is None (the default set in the constructor), the getter method will attempt
         to get the value from the wrapper's output_shape property (if a wrapper is set). Otherwise,
         the getter will return the _output_shape value.

         Parameters
         ----------
         value : n-tuple
             The new input shape value to set.

         """

        self._output_shape = value

    @property
    def input_size(self):
        """
        obtains the number of input elements  of the EmbedIA layer/element,
        regardless of the shape.
        Returns
        -------
        int
            number of input elements
        """

        s = 1
        for d in self.input_shape:
            s *= d
        return s

    @property
    def output_size(self):
        """
        obtains the number of output elements  of the EmbedIA layer/element,
        regardless of the shape.
        Returns
        -------
        int
            number of output elements
        """

        s = 1
        for d in self.output_shape:
            s *= d
        return s

    @property
    def required_files(self):
        '''
        retorna una lista de tuplas indicando los nombres de los archivos donde se encuentra la definicion de
        tipos de datos (.h) y la implementación de las funciones (.c) requeridos por la capa/elemento
        '''
        return [('common.h', 'common.c')]

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        return 0

    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        return 0

    def get_info(self, types_dict):
        """
        Gets info of the layer/object
        Returns
        -------
        LayerInfo object
            information of layer: name, type name, #params, MACs, output shape

        """

        return LayerInfo(self, types_dict)

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be implemented in
        "neural_net.c" and by convention should be called "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras Dense
        layer, the function "dense_layer" must be implemented in "neural_net.c"
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
        return ''

    def debug_function(self, param):
        """
        generates C code with the debug function invocation with the output
        result of the layer to be implemented in the file "embedia_debug.c".
        The default behavior generates an invocation to a function whose name
        is composed of "print_"+"type of data to print" (data1d_t, data2d_t or
        data3d_t).
        Parameters
        ----------
        param : str
            name of the variable to be used in the invocation of the C
            function that implements the debug function.
        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "neural_net.c".
        """
        name = self.name
        dbg_fn = 'print_' + self.output_data_type
        return f'''{dbg_fn}("{name}", {param});'''
