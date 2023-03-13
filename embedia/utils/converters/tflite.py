import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Dense, Conv2D,
                                     SeparableConv2D, MaxPooling2D,
                                     Reshape, AveragePooling2D, LeakyReLU,
                                     Activation, Flatten
                                     )

from tensorflow.lite.python import schema_py_generated as schema_fb

from embedia.utils.string_utils import CamelCaseToSnakeCase, NameGenerator

# def TensorTypeToName(tensor_type):
#   """Converts a numerical enum to a readable tensor type."""
#   for name, value in schema_fb.TensorType.__dict__.items():
#     if value == tensor_type:
#       return name
#   return None


class OpCodeMapper:
    """Maps an opcode index to an op name."""

    def __init__(self, data):
        self._map_data(data)

    def _map_data(self, data):
        self.code_to_name = {}
        for idx, d in enumerate(data.operatorCodes):
            self.code_to_name[idx] = self._builtin_code_to_name(d.builtinCode)
            if self.code_to_name[idx] == "CUSTOM":
                self.code_to_name[idx] = self._name_list_to_str(d.customCode)

    def _builtin_code_to_name(self, code):
        """Converts a builtin op code enum to a readable name."""
        for name, value in schema_fb.BuiltinOperator.__dict__.items():
            if value == code:
                return name
        return None

    def _name_list_to_str(name_list):
        """Converts a list of integers to the equivalent ASCII string."""
        if isinstance(name_list, str):
            return name_list
        else:
            result = ""
            if name_list is not None:
                for val in name_list:
                    result = result + chr(int(val))
        return result

    def get_op_name(self, x):
        if x not in self.code_to_name:
            return "<UNKNOWN>"
        else:
            return self.code_to_name[x]


class TFLiteModelConverter:

    def __init__(self):
        pass

    def set_from_buffer(self, flatbuffer, interpreter):
        interpreter.allocate_tensors()
        self._interpreter = interpreter
        self._fb_model = self._get_flatbuffer_model(flatbuffer)
        self._model = None
        self._opcode_mapper = OpCodeMapper(self._fb_model)

    def _flatbuffer_to_dict(self, fb, preserve_as_numpy):
        """Converts a hierarchy of FB objects into a nested dict.

        We avoid transforming big parts of the flat buffer into python arrays.
        This speeds conversion from ten minutes to a few seconds on big graphs.

        Args:
          fb: a flat buffer structure. (i.e. ModelT)
          preserve_as_numpy: true if all downstream np.arrays should be
            preserved. false if all downstream np.array should become python
            arrays
        Returns:
          A dictionary representing the flatbuffer rather than a flatbuffer
          object
        """
        if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
            return fb
        elif hasattr(fb, "__dict__"):
            result = {}
            for attribute_name in dir(fb):
                attribute = fb.__getattribute__(attribute_name)
                if not callable(attribute) and attribute_name[0] != "_":
                    snake_name = CamelCaseToSnakeCase(attribute_name)
                    preserve = True if attribute_name == "buffers" else preserve_as_numpy
                    result[snake_name] = self._flatbuffer_to_dict(attribute, preserve)
            return result
        elif isinstance(fb, np.ndarray):
            return fb if preserve_as_numpy else fb.tolist()
        elif hasattr(fb, "__len__"):
            return [self._flatbuffer_to_dict(entry, preserve_as_numpy) for entry in fb]
        else:
            return fb

    def _get_flatbuffer_model(self, buffer_data):
        model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
        model = schema_fb.ModelT.InitFromObj(model_obj)
        return model
        #return self._flatbuffer_to_dict(model, preserve_as_numpy=False)

    def load_model(self, filename):
        with open(filename, "rb") as file_handle:
            flatbuffer_array = bytearray(file_handle.read())

        interpreter = tf.lite.Interpreter(model_path=filename)

        self.set_from_buffer(flatbuffer_array, interpreter)

    def _get_weights(self, tensor_id):
        return self._interpreter.tensor(tensor_id)()

    def get_tf_model(self):
        if self._model is None:
            self._convert_model()
        return self._model


    def _convert_model(self):

        self._model = Sequential()
        # print(info)

        name_gen = NameGenerator()

        self.add_input()

        for graph in self._fb_model.subgraphs:

            for operator in graph.operators:
                code = operator.opcodeIndex
                name = self._opcode_mapper.get_op_name(code)
                print(name)
                if name in [ 'MUL', 'ADD']:
                	print(operator)

                if name == 'FULLY_CONNECTED':
                    layer = self.add_dense(operator)
                elif name == 'CONV_2D':
                    layer = self.add_conv2d(operator)
                elif name == 'MAX_POOL_2D':
                    layer = self.add_pool2d(MaxPooling2D, operator)
                elif name == 'AVERAGE_POOL_2D':
                    layer = self.add_pool2d(AveragePooling2D, operator)
                elif name == 'DEPTHWISE_CONV_2D':
                    layer = self.add_separable_conv2d(operator)
                elif name == 'RESHAPE':
                    layer = self.add_reshape(operator)
                elif name == 'LEAKY_RELU':
                    layer = self.add_leakyrelu(operator)
                elif name == 'RELU':
                    layer = self.add_activation(tf.nn.relu)
                elif name == 'EXP':
                    layer = self.add_activation('exponential')
                elif name == 'TANH':
                    layer = self.add_activation(tf.nn.tanh)
                elif name == 'LOGISTIC':
                    layer = self.add_activation(tf.nn.sigmoid)
                elif name == 'SOFTMAX':
                    layer = self.add_activation(tf.nn.softmax)
                elif name == 'LOG_SOFTMAX':
                    layer = self.add_activation(tf.nn.log_softmax)
                else:
                    layer = None
                if layer is not None:
                    layer._name = name_gen.get(name)
        return self._model
    # TO DO: implement PRELU, ELU, SELU ARG_MAX ARG_MIN, gelu? linear? softsign? softplus? swish?

    def add_input(self):
        sg = self._fb_model.subgraphs[0]
        input_index = sg.inputs[0]
        shape = tuple(sg.tensors[input_index].shape[1:])

        self._model.add(InputLayer(input_shape=shape))

    def add_dense(self, operator):
        weights = self._interpreter.tensor(operator.inputs[-2])()
        bias_index = operator.inputs[-1]
        if bias_index >= 0:
            bias = self._interpreter.tensor(bias_index)()
        else:
            bias = np.zeros(len(weights))

        layer = Dense(units=len(bias))

        self._model.add(layer)

        # verify channels first/last of weights
        (ls, ws) = (tuple(layer.weights[0].shape), weights.shape)
        if ls[-1] == ws[0] and ls[0] == ws[-1]:
            weights = np.transpose(weights, (1, 0))

        layer.set_weights([weights, bias])

        return layer

    def add_conv2d(self, operator):
        (opt, inp) = (operator.builtinOptions, operator.inputs)

        bias = self._get_weights(inp[-1])
        weights = self._get_weights(inp[-2])
        weights = np.transpose(weights, (1, 2, 3, 0))

        filters = bias.shape[0]
        kernel_size = (weights.shape[0], weights.shape[1])

        strides = (opt.strideH, opt.strideW)
        # padding = opt['padding'] # "valid" or "same"
        padding = 'valid'  # to do analize padding

        dilation_rate = (opt.dilationHFactor, opt.dilationWFactor)

        layer = Conv2D(filters=filters, kernel_size=kernel_size,
                       strides=strides, padding=padding,
                       data_format='channels_last',
                       dilation_rate=dilation_rate)

        self._model.add(layer)

        layer.set_weights([weights, bias])

        return layer

    def _get_separable_conv2d_weights(self, inputs):
        # bias parameter is missing, search tensor
        bias = None
        name = self._interpreter.get_tensor_details()[inputs[2]]['name'].split('/')
        name = '%s/%s/BiasAdd/' % tuple(name[0:2])
        for tensor in self._interpreter.get_tensor_details():
            if tensor['name'].startswith(name):
                bias = self._get_weights(tensor['index'])
                break
        if bias is None:
            bias = self._get_weights(inputs[2])
        w_dp = np.transpose(self._get_weights(inputs[1]), (1, 2, 3, 0))
        w_pt = np.transpose(self._get_weights(inputs[1]+1), (1, 2, 3, 0))

        return (w_dp, w_pt, bias)

    def add_separable_conv2d(self, operator):
        (opt, inp) = (operator.builtinOptions, operator.inputs)

        w_dp, w_pt, bias = self._get_separable_conv2d_weights(inp)

        filters = w_pt.shape[3]
        kernel_size = (w_dp.shape[0], w_dp.shape[1])

        strides = (opt.strideH, opt.strideW)
        # padding = opt['padding'] # "valid" or "same"
        padding = 'valid'  # to do analize padding

        dilation_rate = (opt.dilationHFactor, opt.dilationWFactor)
        depth_multiplier = opt.depthMultiplier
        layer = SeparableConv2D(filters=filters, kernel_size=kernel_size,
                       strides=strides, padding=padding,
                       data_format='channels_last',
                       dilation_rate=dilation_rate,
                       depth_multiplier=depth_multiplier)

        self._model.add(layer)

        # weights_depth = np.transpose(weights_depth, (1, 2, 3, 0))
        # weights_point = np.transpose(weights_point, (1, 2, 3, 0))

        layer.set_weights([w_dp, w_pt, bias])

        return layer

    def add_pool2d(self, pool_class, operator):

        opt = operator.builtinOptions
        pool_size = (opt.filterHeight, opt.filterWidth)
        strides = (opt.strideH, opt.strideW)
        # padding = opt['padding'] # "valid" or "same"
        padding = 'valid'  # to do analize padding

        layer = pool_class(pool_size=pool_size, strides=strides,
                           padding=padding, data_format='channels_last'
                           )
        self._model.add(layer)

        return layer

    def add_reshape(self, operator):
        layer = Flatten()
        self._model.add(layer)
        return layer

    def add_activation(self, func_name):
        layer = Activation(func_name)
        self._model.add(layer)
        return layer

    def add_leakyrelu(self, operator):
        layer = LeakyReLU(alpha=operator.builtinOptions.alpha)
        self._model.add(layer)
        return layer


def load_model(filename):

    converter = TFLiteModelConverter()
    converter.load_model(filename)

    return converter


def convert_to_tf(converter):
    return converter.get_tf_model()
