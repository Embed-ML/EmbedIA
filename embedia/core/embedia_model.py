from embedia.core.layers_implemented import dict_layers
from embedia.core.layer import EmbediaFile
from collections import defaultdict
from embedia.model_generator.project_options import ModelDataType
import regex as re
from embedia.core.unimplemented_layer import UnimplementedLayer
from embedia.core.type_converters import *
from embedia.core.exceptions import *
from embedia.core.layer_wrapper import OutputPredictionType
from collections.abc import Iterable

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
        model_name = self._model.name.lower()
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
            return [(EmbediaFile('fixed.h'), EmbediaFile('fixed.c'))]
        elif data_type == ModelDataType.FIXED16:
            return [(EmbediaFile('fixed.h'), EmbediaFile('fixed.c'))]
        elif data_type == ModelDataType.FIXED32:
            return [(EmbediaFile('fixed.h'), EmbediaFile('fixed.c'))]
        elif data_type == ModelDataType.QUANT8:
            return [(EmbediaFile('quant8.h'), EmbediaFile('quant8.c')), (EmbediaFile('fixed.h'), EmbediaFile('fixed.c'))]
        elif data_type == ModelDataType.FULL_QUANT8:
            return [(EmbediaFile('quant8.h'), EmbediaFile('quant8.c'))]
        elif data_type == ModelDataType.BINARY:
            return None
        elif data_type == ModelDataType.BINARY_FIXED32:
            return [(EmbediaFile('fixed.h'), EmbediaFile('fixed.c'))]
        elif data_type == ModelDataType.BINARY_FLOAT16:
            return [(EmbediaFile('half.hpp'), None)]
        return None

    @property
    def required_files(self):
        result = []
        dt_files = self._get_data_type_files()
        if dt_files is not None:
            for dt_file in dt_files:
                if dt_file not in result:
                    result.append(dt_file)
        for layer in self._embedia_layers:
            for files_tuple in layer.required_files:
                if files_tuple not in result:
                    result.append(files_tuple)
        return result

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

    from collections.abc import Iterable

    def _extract_layers(self, model):
        """
        Normaliza cualquier tipo de modelo a una lista de 'bloques procesables'.

        Soporta:
        - Keras models (model.layers)
        - sklearn ensembles (estimators_)
        - DummyModel (layers)
        - listas/tuplas de capas
        - single objects (wrap en lista)
        """

        # 1. Keras / DummyModel
        if hasattr(model, 'layers'):
            return list(model.layers)

        # 2. sklearn ensembles (RandomForest, etc.)
        if hasattr(model, 'estimators_'):
            return list(model.estimators_)

        # 3. sklearn pipelines (opcional, si lo usás)
        if hasattr(model, 'steps'):
            return [step[1] for step in model.steps]

        # 4. iterable explícito (pero no string)
        if isinstance(model, Iterable) and not isinstance(model, (str, bytes)):
            return list(model)

        # 5. fallback → objeto único
        return [model]

    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # external preprocessing to the model? => add as first layer
        self._add_processing_layers(self.options.preprocessing)

        for layer in self._extract_layers(self.model):
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

    def get_type_converter(self, data_type=None, fixed_precision=None):
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
            return ('quant8', QuantizedTypeConverter(8, symetric=False, signed=True))
        elif data_type == ModelDataType.FULL_QUANT8:
            return ('quant8', QuantizedTypeConverter(8, symetric=False, signed=True))
        elif data_type.is_fixed_point:
            if fixed_precision is not None:
                fraq_bits = fixed_precision
            elif self.options.fixed_precision is not None:
                fraq_bits = self.options.fixed_precision
            elif data_type == ModelDataType.FIXED32:
                fraq_bits = 17
            elif data_type == ModelDataType.FIXED16:
                fraq_bits = 8
            else:
                frac_bits = 4
            int_bits = data_type.size - fraq_bits
            return ('fixed', FixedTypeConverter(int_bits, fraq_bits))
        else:
            raise UnsupportedFeatureError(data_type, 'Data type converter not supported')

    @property
    def is_data_quantized(self):
        """
        Check if the data is quantized.

        Returns:
            bool: True if the data is quantized (ModelDataType.QUANT8), False otherwise.
        """
        return self.options.data_type.is_quantized

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

    def get_buffer_layer_size(self, layer_idx, align=0):
        """
        Buffer size for a specific layer — used for layer statistics.

        Returns the size of the largest slot this layer uses, which is
        max(input_slot, output_slot + internal_temp).
        This represents the heaviest buffer this layer handles at any point.

        For the total RAM required at runtime (both slots simultaneously),
        use _get_layer_working_size().

        Args:
            layer_idx : index of the layer
            align     : alignment in bytes (0 = no alignment)

        Returns:
            int: largest slot size in bytes (0 for DummyLayers)
        """

        def ensure_aligned(size, align):
            if align > 0:
                size = ((size + align - 1) // align) * align
            return size

        layer = self.embedia_layers[layer_idx]

        if layer.__class__.__name__ == 'DummyLayer':
            return 0

        if self.options.data_type == ModelDataType.QUANT8:
            data_type_sz = 16 // 8
        else:
            data_type_sz = self.options.data_type.size // 8

        is_first_real = all(
            self.embedia_layers[i].__class__.__name__ == 'DummyLayer'
            for i in range(layer_idx)
        )

        inp_sz = ensure_aligned(layer.input_size * data_type_sz, align)
        out_sz = ensure_aligned(layer.output_size * data_type_sz, align)
        int_extra = ensure_aligned(layer.internal_alloc_required, align)

        if align > 0 and layer.internal_alloc_required > 0:
            int_extra += 3 # +3 cubre peor caso de padding entre slices internos (swap_alloc_slice3)

        if layer.inplace_output: # capas inplace no requieren memoria salvo la primera
            if is_first_real: # primer capa real es inplace, requiere copiar la salida
                return ensure_aligned(out_sz + int_extra, align)
            return 0
            #return ensure_aligned(max(inp_sz, out_sz) + int_extra, align)
        else:
            # el slot más grande entre input y (output+temp)
            return ensure_aligned(max(inp_sz, out_sz + int_extra), align)

    def get_buffer_layer_max_size(self, align=0):
        """
        Returns ALLOC_BUFFER_SZ — maximum working size across all layers.
        """
        if not self.embedia_layers:
            return 0
        return max(
            self._get_layer_working_size(i, align)
            for i in range(len(self.embedia_layers))
        )


    def _get_layer_working_size(self, layer_idx, align=0):
        """
        Calculates the total pool size required for a specific layer.
        Pool = slot_A + slot_B, both must fit simultaneously in ALLOC_BUFFER_SZ.

        Args:
            layer_idx : index of the layer
            align     : alignment in bytes (0 = no alignment)

        Returns:
            int: total pool size in bytes (0 for DummyLayers)
        """

        def ensure_aligned(size, align):
            if align > 0:
                size = ((size + align - 1) // align) * align
            return size

        layer = self.embedia_layers[layer_idx]

        if layer.__class__.__name__ == 'DummyLayer':
            return 0

        if self.options.data_type == ModelDataType.QUANT8:
            data_type_sz = 16 // 8
        else:
            data_type_sz = self.options.data_type.size // 8

        is_first_real = all(
            self.embedia_layers[i].__class__.__name__ == 'DummyLayer'
            for i in range(layer_idx)
        )

        out_sz = ensure_aligned(layer.output_size * data_type_sz, align)

        # +3 cubre peor caso de padding entre slices internos (swap_alloc_slice3)
        int_extra = ensure_aligned(layer.internal_alloc_required, align)
        if align > 0 and layer.internal_alloc_required > 0:
            int_extra += 3

        # primera capa real — input externo, pool = output + temp
        if is_first_real:
            return ensure_aligned(out_sz + int_extra, align)

        inp_sz = ensure_aligned(layer.input_size * data_type_sz, align)

        if layer.inplace_output:
            # inplace — un buffer efectivo, tamaño = max(inp, out) + temp
            return ensure_aligned(max(inp_sz, out_sz) + int_extra, align)
        else:
            # doble buffer:
            #   slot A (izq) = inp_sz
            #   slot B (der) = out_sz + int_extra  (swap_alloc_slice)
            #   pool = A + B
            return ensure_aligned(inp_sz + out_sz + int_extra, align)

    def get_buffer_layer_max_size(self, align=0):
        """
        Returns the value for ALLOC_BUFFER_SZ — the total static pool size
        required to run inference for any layer in the model.

        This is the only externally used method for buffer size calculation.

        Args:
            align: alignment in bytes (0 = no alignment)

        Returns:
            int: maximum pool size needed across all layers
        """
        if not self.embedia_layers:
            return 0

        return max(
            self._get_layer_working_size(i, align)
            for i in range(len(self.embedia_layers))
        )

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

