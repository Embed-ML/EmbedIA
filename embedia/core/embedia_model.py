from embedia.core.layers_implemented import dict_layers
from collections import defaultdict
from embedia.model_generator.project_options import ModelDataType
import regex as re
import pycparser as pcp
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.core.unimplemented_layer import UnimplementedLayer
from embedia.core.type_converters import *
from embedia.core.exceptions import *


class EmbediaModel(object):
    """
    Class for representing and managing an EmbedIA model.

    This class provides functionality for:
    - Initializing a model from a TensorFlow/SkLearn/other model object
    - Managing model layers/modules/components
    - Handling data types and model options
    - Generating information about model layers
    """

    def __init__(self, obj_model, options):
        """
         Initializes an EmbediaModel object.

         Args:
             obj_model: The TensorFlow/SkLearn/Other model object to be converted.
             options: Project-specific options for model generation.
         """

        self._types_dict = {}
        self._options = None
        self._embedia_layers = []

        self._options = options
        self._clear_names()
        self.model = obj_model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, a_model):
        self._model = a_model
        self._create_embedia_layers()

    @property
    def embedia_layers(self):
        return self._embedia_layers

    @property
    def types_dict(self):
        return self._types_dict

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        self._options = options

    def _get_data_type_file(self):
        data_type = self.options.data_type
        if data_type == ModelDataType.FIXED8:
            return  ('fixed.h', 'fixed.c')
        elif data_type == ModelDataType.FIXED16:
            return  ('fixed.h', 'fixed.c')
        elif data_type == ModelDataType.FIXED32:
            return  ('fixed.h', 'fixed.c')
        elif data_type == ModelDataType.QUANT8:
            return ('quant8.h', 'quant8.c')
        elif data_type == ModelDataType.BINARY:
            return None
        elif data_type == ModelDataType.BINARY_FIXED32:
            return ('fixed.h', 'fixed.c')
        elif data_type == ModelDataType.BINARY_FLOAT16:
            return ('half.hpp', None)
        return None

    @property
    def required_files(self):
        result = set()
        dt_files = self._get_data_type_file()
        if dt_files is not None:
            result.add(dt_files)
        for layer in self._embedia_layers:
            for files_tuple in layer.required_files:
                result.add(files_tuple)
        return list(result)


    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # external normalizer to the model? => add as first layer
        if self.options.normalizer is not None:
            obj = self.options.normalizer
            ly = self._create_embedia_layer(obj)
            self._embedia_layers.append(ly)

        for layer in self.model.layers:
            obj = layer
            ly = self._create_embedia_layer(layer)
            self._embedia_layers.append(ly)

        self._complete_layers_shapes()

        return self.embedia_layers

    def _create_embedia_layer(self, obj):
        """
        Create an EmbediA layer from an object.

        Parameters:
            obj (object): The object from which the EmbediA layer will be created.

        Returns:
            EmbediaLayer: The created EmbediA layer.
        """
        try:
            (layer_class, wrapper_class) = dict_layers[type(obj)]
            if wrapper_class is None:
                wrapper = None
            else:
                wrapper = wrapper_class(obj)
            layer = layer_class(self, wrapper)
        except KeyError:
            layer = UnimplementedLayer(self, obj)
        return layer

    def _complete_layers_shapes(self):

        for layer in self.embedia_layers:
            if layer.input_shape is None:
                layer.input_shape = self.get_input_shape(layer)
            if layer.output_shape is None:
                layer.output_shape = self.get_output_shape(layer)

    def get_input_shape(self, embedia_layer):
        try:
            idx = self.embedia_layers.index(embedia_layer)
        except ValueError:
            return None

        # Check previous layers
        for i in range(idx - 1, -1, -1):
            if self.embedia_layers[i].output_shape is not None:
                return self.embedia_layers[i].output_shape
            elif self.embedia_layers[i].input_shape is not None:
                return self.embedia_layers[i].input_shape

        # Check following layers
        for i in range(idx + 1, len(self.embedia_layers)):
            if self.embedia_layers[i].input_shape is not None:
                return self.embedia_layers[i].input_shape
            elif self.embedia_layers[i].output_shape is not None:
                return self.embedia_layers[i].output_shape

        return None

    def get_output_shape(self, embedia_layer):
        try:
            idx = self.embedia_layers.index(embedia_layer)
        except ValueError:
            return None

        # Check following layers
        for i in range(idx + 1, len(self.embedia_layers)):
            if self.embedia_layers[i].input_shape is not None:
                return self.embedia_layers[i].input_shape
            elif self.embedia_layers[i].output_shape is not None:
                return self.embedia_layers[i].output_shape

        # Check previous layers
        for i in range(idx - 1, -1, -1):
            if self.embedia_layers[i].output_shape is not None:
                return self.embedia_layers[i].output_shape
            elif self.embedia_layers[i].input_shape is not None:
                return self.embedia_layers[i].input_shape

        return None


    def _clear_names(self):
        self.names = defaultdict(lambda: 0)

    def get_unique_name(self, obj):
        """
        Generate a unique name for the given object.

        Parameters:
            obj (object) or string: The object or string for which to generate a unique name.

        Returns:
            str: A unique name for the object.
        """
        if type(obj) == str:
            name = obj
        else:
            if obj.wrapper is not None:
                obj = obj.wrapper
                if hasattr(obj, "name"):
                    name = obj.name
                elif hasattr(obj, "__name__"):
                    name = obj.__name__
                else:
                    name = obj.__class__.__name__
            else:
                name = obj.__class__.__name__

        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        num = self.names[name]
        self.names[name] += 1
        if num == 0:
            return name
        return name+str(num)

    def get_type_converter(self, data_type=None):
        """
        returns a tuple with the name of the embedia type used (float, fixed, quant8) in the
        data representation (e.g. neuron weights) together with the conversion
        object to be invoked to transform a float value to the data type

        Parameters
        ----------
        data_type : ModelDataType
            variable with the data type used in the data representation
            (float, fixed8, fixed16, fixed32, quant8, etc)

        Returns
        -------
        tuple (str, TypeConverter object)
            tuple with type and macro convertion for C.

        """

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
        if data_type is None:
            data_type = self.options.data_type
        if data_type == ModelDataType.FLOAT:
            return ('float', FloatConverter())
        elif data_type == ModelDataType.BINARY: # binary layers dont use data_type, use block_type
            return ('float', FloatConverter())  # should use FloatConverter()?, test required
        elif data_type == ModelDataType.BINARY_FLOAT16:
            return ('half', FloatConverter()) # should use Float16TypeConverter(), but not suppoerted in other layers?, test required
        elif data_type == ModelDataType.BINARY_FIXED32:
            return ('fixed', FixedTypeConverter(15, 17))  # should use Fixed32TypeConverter()?, test required
        elif data_type == ModelDataType.QUANT8:
            return ('quant8', QuantizedTypeConverter(8, False))
        elif data_type == ModelDataType.FIXED32:
            return ('fixed', FixedTypeConverter(15, 17))
        elif data_type == ModelDataType.FIXED16:
            return ('fixed', FixedTypeConverter(8, 8))
        elif data_type == ModelDataType.FIXED8:
            return ('fixed', FixedTypeConverter(5, 3))
        else:
            raise UnsupportedFeatureError(data_type, 'Data type converter not supported')

    def identify_target_classes(self):
        """
        Identify the number of target classes based on the last layer of the model.

        Returns:
            int: 1 for binary classification, N for multiclass classification with N classes,
                or 0 for regression.
        """
        wrapper = self.embedia_layers[-1].wrapper
        act_fn = ''
        if wrapper.activation is not None:
            act_fn = wrapper.activation.__name__.lower()

        if act_fn == '':
            act_fn = wrapper.target.__class__.__name__.lower()

        if act_fn in ['sigmoid', 'sigmoidal', 'softsign', 'tanh']:
                return 1 # binary classification
        if act_fn == 'softmax':
            return wrapper.output_shape[-1] # multiclass classification

        return 0 # regression

    @property
    def is_data_quantized(self):
        """
        Check if the data is quantized.

        Returns:
            bool: True if the data is quantized (ModelDataType.QUANT8), False otherwise.
        """
        return self.options.data_type == ModelDataType.QUANT8

    def get_type_initializer(self):
        """
        Returns a function whose purpose is to explore the data to obtain conversion parameters, such as in the case
        of 8-bit quantization.

        Returns
        -------
        function(data)
            function to explore data to extract parameters for convertion

        """

        if self.options.data_type == ModelDataType.QUANT8:
            def data_type_explorer(values):
                Q_MAX = 255
                min_val = np.min(values)
                max_val = np.max(values)
                # Calcular la escala y el punto cero para la cuantización
                scale = (max_val - min_val) / Q_MAX
                zero_pt = -min_val / scale  # Punto cero para mapear al rango

                if zero_pt < 0:
                    zero_pt = 0
                elif zero_pt > Q_MAX:
                    zero_pt = Q_MAX
                else:
                    zero_pt = round(zero_pt)
                return (scale, zero_pt)
        else:
            def data_type_explorer(values):
                return None

        return data_type_explorer



    def _build_types_size_dict(self, embedia_decl):
        """
        Build a dictionary of types and their sizes in bytes from the EmbediA declaration.

        Parameters:
           embedia_decl (str): The EmbediA declaration.

        Returns:
           dict: Dictionary of types and their sizes in bytes.
        """

        # prepare to extract declaration of structures
        # get code to first function definition in order to includes structures
        if (self.options.data_type == ModelDataType.BINARY or self.options.data_type == ModelDataType.BINARY_FIXED32 or self.options.data_type == ModelDataType.BINARY_FLOAT16):
            start = embedia_decl.find('endif')
            start = start + 5
        else:
            start = embedia_decl.find('typedef')
        end = embedia_decl.find('void')
        embedia_decl = embedia_decl[start:end]

         # remove comments, pycparser doesnt support them
        pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
        embedia_decl = re.sub(pattern, '', embedia_decl)

        # add dummy base data type declaration into EmbedIA code
        embedia_decl = """
typedef char uint8_t;
typedef char uint16_t;
typedef char uint32_t;
typedef char uint64_t;
typedef char xBITS;
typedef char fixed;
typedef char dfixed;
typedef char half;
typedef char quant8;
typedef char qparam_t;
typedef char size2d_t;
""" + embedia_decl

        # delete lines with '#' because parser doesnt support
        embedia_decl = re.sub(r'^\s*#.*\n', '', embedia_decl, flags=re.MULTILINE)
        parser = pcp.CParser()

        code = parser.parse(embedia_decl)
        bytes_size4 = 2
        if(self.options.tamano_bloque==BinaryBlockSize.Bits8):
            bytes_size = 1
        elif(self.options.tamano_bloque==BinaryBlockSize.Bits16):
            bytes_size = 2
        elif(self.options.tamano_bloque==BinaryBlockSize.Bits32):
            bytes_size = 4
        else:
            bytes_size = 8
        
        bytes_size2 = 0
        bytes_size3 = 0
        if(self.options.data_type == ModelDataType.FIXED8):
            bytes_size2 = 1
            bytes_size3 = 2
        elif(self.options.data_type == ModelDataType.FIXED16):
            bytes_size2 = 2
            bytes_size3 = 4
        elif(self.options.data_type == ModelDataType.FIXED32 or self.options.data_type == ModelDataType.BINARY_FIXED32):
            bytes_size2 = 4
            bytes_size3 = 8
        # base types sizes in bytes
        self._types_dict = {
            'uint8_t': 1,
            'uint16_t': 2,
            'uint32_t': 4,
            'uint64_t': 8,
            'float': 4,
            'quant8': 1,
            'qparam_t': 5,
            'size2d_t': 4,
            'xBITS': bytes_size,
            'fixed': bytes_size2,
            'dfixed': bytes_size3,
            'half': bytes_size4
        }

        for node in code.ext:
            if type(node) is pcp.c_ast.Typedef and not node.name in  self._types_dict:
                dt_type = node.name
                dt_size = self._explore_type(node)
                self._types_dict[dt_type] = dt_size

        return self._types_dict

    def _explore_type(self, node):
        """
        Explore the size of a data type.

        Parameters:
            node (object): The abstract syntax tree node representing the data type.

        Returns:
            int: The size in bytes of the data type.
        """
        if type(node.type.type) is pcp.c_ast.Struct:
            size = 0
            for d in node.type.type.decls:
                size += self._explore_type(d)
            return size
        elif type(node.type) is pcp.c_ast.PtrDecl:
            return 4  # always return 4 bytes for now
        elif node.type.type.names[0] in self._types_dict:
            return self._types_dict[node.type.type.names[0]]

        raise Exception(node.type)

    def get_layers_info(self, embedia_decl):
        """
        Get the information of the model's layers.

        Parameters:
          embedia_decl (str): The EmbediA declaration.

        Returns:
          list: List of tuples with information for each layer (name, type, activation, parameters,
                output shape, MACs, memory size).
        """

        if len(self._types_dict) == 0:
            self._build_types_size_dict(embedia_decl)

        layers_info = []
        for layer in self.embedia_layers:
            info = layer.get_info(self._types_dict)
            l_type = info.class_name
            l_name = info.layer_name
            params = info.params
            shape = info.output_shape
            MACs = info.macs_ops
            size = info.memory

            layers_info.append((l_name, l_type, params, shape, MACs, size))

        return layers_info

    def firstLayerOfItsclass(self, embedia_layer):
        """
         Check if the given EmbediA layer is the first layer of its class in the model.

         Parameters:
             embedia_layer (EmbediaLayer): The EmbediA layer to check.

         Returns:
             bool: True if the layer is the first of its class, False otherwise.
         """
        for layer in self.embedia_layers:
            if type(embedia_layer) is type(layer):
                return embedia_layer == layer
        return False

