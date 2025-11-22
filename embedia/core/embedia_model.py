from embedia.core.layers_implemented import dict_layers
from collections import defaultdict
from embedia.model_generator.project_options import ModelDataType
import regex as re
from embedia.core.unimplemented_layer import UnimplementedLayer
from embedia.core.type_converters import *
from embedia.core.exceptions import *
from embedia.core.layer_wrapper import OutputPredictionType
from embedia.utils import string_utils
from enum import Enum

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
        self._output_prediction_type = None

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

    @property
    def model_name(self):
        model = self._model
        if hasattr(model, 'name'):
            model_name = model.name
        else:
            model_name = string_utils.CamelCaseToSnakeCase(type(model).__name__)

        model_name = model_name.lower()

        if not model_name.endswith('model'):
            model_name += '_model'
        return model_name

    @property
    def output_prediction_type(self):
        if self._output_prediction_type is None:
            self._output_prediction_type = self._infer_output_prediction_type()

        return self._output_prediction_type

    def _infer_output_prediction_type(self):
        return OutputPredictionType.CLASS_PROBABILITIES

    def _get_data_type_files(self):
        data_type = self.options.data_type
        if data_type == ModelDataType.FIXED8:
            return  [('fixed.h', 'fixed.c')]
        elif data_type == ModelDataType.FIXED16:
            return  [('fixed.h', 'fixed.c')]
        elif data_type == ModelDataType.FIXED32:
            return  [('fixed.h', 'fixed.c')]
        elif data_type == ModelDataType.QUANT8:
            return [('quant8.h', 'quant8.c')]
        elif data_type == ModelDataType.FULL_QUANT8:
            return [('quant8.h', 'quant8.c'), ('fixed.h', 'fixed.c')]
        elif data_type == ModelDataType.BINARY:
            return None
        elif data_type == ModelDataType.BINARY_FIXED32:
            return [('fixed.h', 'fixed.c')]
        elif data_type == ModelDataType.BINARY_FLOAT16:
            return [('half.hpp', None)]
        return None

    @property
    def required_files(self):
        result = set()
        dt_files = self._get_data_type_files()
        if dt_files is not None:
            for dt_file in dt_files:
                result.add(dt_file)
        for layer in self._embedia_layers:
            for files_tuple in layer.required_files:
                result.add(files_tuple)
        return list(result)

    def _add_processing_layers(self, objects):
        if objects is None:
            return

        # Convertir a iterable (manejando diccionarios, strings y objetos no iterables)
        if isinstance(objects, dict):
            objects = objects.values()  # Usa la vista de valores (sin copiar)
        elif not hasattr(objects, '__iter__'):
            objects = [objects]  # Non iterable objects

        for object in objects:
            ly = self._create_embedia_layer(object)
            self._embedia_layers.append(ly)


    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # external preprocessing to the model? => add as first layer
        self._add_processing_layers(self.options.preprocessing)

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

        name = re.sub(r'(?<=[a-zA-Z])(?=[A-Z][a-z])', '_', name)
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
        elif data_type == ModelDataType.FULL_QUANT8:
            return ('quant8', QuantizedTypeConverter(8, symetric=False, signed=True))
        elif data_type == ModelDataType.FIXED32:
            return ('fixed', FixedTypeConverter(15, 17))
        elif data_type == ModelDataType.FIXED16:
            return ('fixed', FixedTypeConverter(8, 8))
        elif data_type == ModelDataType.FIXED8:
            return ('fixed', FixedTypeConverter(5, 3))
        else:
            raise UnsupportedFeatureError(data_type, 'Data type converter not supported')

    @property
    def is_data_quantized(self):
        """
        Check if the data is quantized.

        Returns:
            bool: True if the data is quantized (ModelDataType.QUANT8), False otherwise.
        """
        return self.options.data_type in [ModelDataType.QUANT8, ModelDataType.FULL_QUANT8]

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

    def get_buffer_layer_max_size(self, align=0):
        """
        Calculates the required buffer size for all layers by finding the maximum
        layer size considering memory alignment. The input size of the first layer
        is excluded as it's pre-allocated.
        """
        if not self.embedia_layers:
            return 0

        # Calculate for all layers and get the maximum
        return max( self.get_buffer_layer_size(i, align) for i in range(len(self.embedia_layers)) )

    def get_buffer_layer_size(self, layer_idx, align=0):
        """
        Calculates the required buffer size for a specific layer considering alignment.
        Args:
            layer_idx: Index of the layer to calculate
            align: Alignment requirement (0 means no alignment)
        Returns:
            int: The calculated size for this specific layer
        """

        def ensure_aligned(size, align):
            """Helper to align size to specified boundary"""
            return ((size + align - 1) // align) * align if align > 0 else size

        data_type_sz = self.options.data_type.size // 8
        layer = self.embedia_layers[layer_idx]

        # First layer only needs output buffer (input is pre-allocated)
        if layer_idx == 0:
            return ensure_aligned(layer.output_size * data_type_sz, align)

        # Other layers
        inp_sz = ensure_aligned(layer.input_size * data_type_sz, align)
        out_sz = ensure_aligned(layer.output_size * data_type_sz, align)

        return max(inp_sz, out_sz) if layer.inplace_output else (inp_sz + out_sz)


    def get_layers_info(self):
        """
        Get the information of the model's layers.

        Parameters:
          embedia_decl (str): The EmbediA declaration.

        Returns:
          list: List of tuples with information for each layer (name, type, activation, parameters,
                output shape, MACs, memory size).
        """

        layers_info = []
        for layer in self.embedia_layers:
            info = layer.get_info(self._types_dict)
            l_type = info.class_name
            l_name = info.layer_name
            params = info.params
            shape = info.output_shape
            MACs = info.macs_ops
            ACOPs= info.ac_ops
            size = info.memory

            layers_info.append((l_name, l_type, params, shape, MACs, ACOPs, size))

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

