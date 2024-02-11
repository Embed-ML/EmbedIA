from embedia.layers.layers_implemented import dict_layers
from collections import defaultdict
from embedia.model_generator.project_options import ModelDataType
import regex as re
import pycparser as pcp
import numpy as np
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.layers.unimplemented_layer import UnimplementedLayer
from embedia.layers.type_converters import *
from embedia.layers.exceptions import *
import tensorflow as tf



class Model(object):
    types_dict = {}

    def __init__(self, options):
        self.options = options
        self.clear_names()


    def set_layers(self, layers, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        embedia_layers = []

        # external normalizar to the model? => add as first layer
        if self.options.normalizer is not None:
            obj = self.options.normalizer
            ly = self.create_embedia_layer(obj)
            embedia_layers.append(ly)

        for layer in layers:
            obj = layer
            ly = self.create_embedia_layer(layer)
            embedia_layers.append(ly)

        self.embedia_layers = embedia_layers
        return embedia_layers

    def create_embedia_layer(self, obj):
        try:
            layer = dict_layers[type(obj)](self, obj, self.options)
        except KeyError:
            layer = UnimplementedLayer(self, obj, self.options)
        return layer

    def get_previous_layer(self, layer):
        try:
            idx = self.embedia_layers.index(layer)
        except ValueError:
            return None
        if idx == 0:
            return None
        return self.embedia_layers[idx-1]


    def clear_names(self):
        self.names = defaultdict(lambda: 0)

    def get_unique_name(self, obj):
        if hasattr(obj, "name"):
            name = obj.name
        elif hasattr(obj, "__name__"):
            name = obj.__name__
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
            if data_type == ModelDataType.FLOAT or data_type == ModelDataType.BINARY:
                return ('float', FloatConverter()) # binary layers dont use data_type, use block_type
            elif data_type == ModelDataType.BINARY_FLOAT16:
                return ('half', None) # should use Float16TypeConverter(), test required
                # def macro_converter(s):
                #    return f"half({s})"
                # data_type = 'half'
            elif data_type == ModelDataType.QUANT8:
                return ('quant8', QuantizedTypeConverter(8, False))
            elif data_type == ModelDataType.FIXED32:
                return ('fixed', FixedTypeConverter(17, 15))
            elif data_type == ModelDataType.FIXED16:
                return ('fixed', FixedTypeConverter(9, 7))
            elif data_type == ModelDataType.FIXED8:
                return ('fixed', FixedTypeConverter(4, 4))
            else:
                raise UnsupportedFeatureError(data_type, 'Data type converter not supported')

    def identify_target_classes(self):
        layer = self.embedia_layers[-1].layer
        act_fn = ''
        if hasattr(layer, 'activation') and layer.activation is not None:
            act_fn = layer.activation.__name__.lower()

        if act_fn == '':
            act_fn = layer.__class__.__name__.lower()

        if act_fn in ['sigmoid', 'sigmoidal', 'softsign', 'tanh']:
                return 1 # binary classification
        if act_fn == 'softmax':
            return layer.output_shape[-1] # multiclass classification

        return 0 # regression

    def is_data_quantized(self):
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
""" + embedia_decl

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
        self.types_dict = {
            'uint8_t': 1,
            'uint16_t': 2,
            'uint32_t': 4,
            'uint64_t': 8,
            'float': 4,
            'quant8': 1,
            'qparam_t': 5,
            'xBITS': bytes_size,
            'fixed': bytes_size2,
            'dfixed': bytes_size3,
            'half': bytes_size4
        }

        for node in code.ext:
            if type(node) is pcp.c_ast.Typedef and not node.name in  self.types_dict:
                dt_type = node.name
                dt_size = self._explore_type(node)
                self.types_dict[dt_type] = dt_size

        return self.types_dict

    def _explore_type(self, node):
        if type(node.type.type) is pcp.c_ast.Struct:
            size = 0
            for d in node.type.type.decls:
                size += self._explore_type(d)
            return size
        elif type(node.type) is pcp.c_ast.PtrDecl:
            return 4  # always return 4 bytes for now
        elif node.type.type.names[0] in self.types_dict:
            return self.types_dict[node.type.type.names[0]]

        raise Exception(node.type)

    def get_layers_info(self, embedia_decl):
        if len(self.types_dict) == 0:
            self._build_types_size_dict(embedia_decl)

        layers_info = []
        for layer in self.embedia_layers:
            info = layer.get_info(self.types_dict)
            l_type = info.class_name
            l_name = info.layer_name
            l_act = info.activation
            params = info.params
            shape = info.output_shape
            MACs = info.macs_ops
            size = info.memory

            layers_info.append((l_name, l_type, l_act, params, shape, MACs, size))

        return layers_info

    def firstLayerOfItsclass(self, embedia_layer):
        for layer in self.embedia_layers:
            if type(embedia_layer) is type(layer):
                return embedia_layer == layer
        return False