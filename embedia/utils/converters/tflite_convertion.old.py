import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Dense, Conv2D, MaxPooling2D,
    Reshape, AveragePooling2D, LeakyReLU, Activation, Flatten
    )
from collections import defaultdict

from tensorflow.lite.python import schema_py_generated as schema_fb

from embedia.utils.string_utils import CamelCaseToSnakeCase, NameGenerator

# def TensorTypeToName(tensor_type):
#   """Converts a numerical enum to a readable tensor type."""
#   for name, value in schema_fb.TensorType.__dict__.items():
#     if value == tensor_type:
#       return name
#   return None


def BuiltinCodeToName(code):
    """Converts a builtin op code enum to a readable name."""
    for name, value in schema_fb.BuiltinOperator.__dict__.items():
        if value == code:
            return name
    return None


def NameListToString(name_list):
    """Converts a list of integers to the equivalent ASCII string."""
    if isinstance(name_list, str):
        return name_list
    else:
        result = ""
        if name_list is not None:
            for val in name_list:
                result = result + chr(int(val))
    return result


class OpCodeMapper:
    """Maps an opcode index to an op name."""

    def __init__(self, data):
        self.map_data(data)

    def map_data(self, data):
        self.code_to_name = {}
        for idx, d in enumerate(data["operator_codes"]):
            self.code_to_name[idx] = BuiltinCodeToName(d["builtin_code"])
            if self.code_to_name[idx] == "CUSTOM":
                self.code_to_name[idx] = NameListToString(d["custom_code"])

    def __call__(self, x):
        if x not in self.code_to_name:
            return "<UNKNOWN>"
        else:
            return self.code_to_name[x]


class DataSizeMapper:
    """For buffers, report the number of bytes."""

    def __call__(self, x):
        if x is not None:
            return "%d bytes" % len(x)
        else:
            return "--"



def FlatbufferToDict(fb, preserve_as_numpy):
    """Converts a hierarchy of FB objects into a nested dict.

    We avoid transforming big parts of the flat buffer into python arrays. This
    speeds conversion from ten minutes to a few seconds on big graphs.

    Args:
      fb: a flat buffer structure. (i.e. ModelT)
      preserve_as_numpy: true if all downstream np.arrays should be preserved.
        false if all downstream np.array should become python arrays
    Returns:
      A dictionary representing the flatbuffer rather than a flatbuffer object.
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
                result[snake_name] = FlatbufferToDict(attribute, preserve)
        return result
    elif isinstance(fb, np.ndarray):
        return fb if preserve_as_numpy else fb.tolist()
    elif hasattr(fb, "__len__"):
        return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
    else:
        return fb


def CreateDictFromFlatbuffer(buffer_data):
    model_obj = schema_fb.TensorflowModel.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return FlatbufferToDict(model, preserve_as_numpy=False)


class TFLiteModel:

    def __init__(self):
        pass

    def set_from_buffer(self, flatbuffer, interpreter):
        self._flatbuffer = flatbuffer
        self._iterpreter = interpreter

    def _build(self):
        # property
        pass

def load_tflite_model(filename):
    with open(filename, "rb") as file_handle:
        flatbuffer_array = bytearray(file_handle.read())

    structure_info = CreateDictFromFlatbuffer(flatbuffer_array)

    tflite_interpreter =tf.lite.Interpreter(model_path=filename)
    tflite_interpreter.allocate_tensors()

    return (structure_info, tflite_interpreter)


def convert_tflite_to_tf(info, interpreter):


    model = Sequential()
    # print(info)
    opcode_mapper = OpCodeMapper(info)
    name_gen = NameGenerator()

    print(info.keys())
    print(info['description'])

    add_input(model, info)

    for graph in info["subgraphs"]:

        for operator in graph["operators"]:
            code = operator['opcode_index']
            name = opcode_mapper(code)
            print(operator)

            if name == 'FULLY_CONNECTED':
                layer = add_dense(model, operator, info, interpreter)
            elif name == 'CONV_2D':
                layer = add_conv2d(model, operator, info, interpreter)
            elif name == 'MAX_POOL_2D':
                layer = add_pool2d(MaxPooling2D, model, operator, info, interpreter)
            elif name == 'AVERAGE_POOL_2D':
                layer = add_pool2d(AveragePooling2D, model, operator, info, interpreter)
            elif name == 'DEPTHWISE_CONV_2D':
                layer = add_deepwise_conv2d(model, operator, info, interpreter)
            elif name == 'RESHAPE':
                layer = add_reshape(model, operator, info, interpreter)
            elif name == 'LEAKY_RELU':
                layer = add_leakyrelu(model, operator, info, interpreter)
            elif name == 'RELU':
                layer = add_activation(tf.nn.relu, model)
            elif name == 'EXP':
                layer = add_activation('exponential', model)
            elif name == 'TANH':
                layer = add_activation(tf.nn.tanh, model)
            elif name == 'LOGISTIC':
                layer = add_activation(tf.nn.sigmoid, model)
            elif name == 'SOFTMAX':
                layer = add_activation(tf.nn.softmax, model)
            elif name == 'LOG_SOFTMAX':
                layer = add_activation(tf.nn.log_softmax, model)
            else:
                layer = None
            if layer is not None:
                layer._name = name_gen.get(name)
    return model
# TO DO: implement PRELU, ELU, SELU ARG_MAX ARG_MIN, gelu? linear? softsign? softplus? swish?


def add_input(model, info):
    input_index = info['subgraphs'][0]['inputs'][0]
    shape = tuple(info['subgraphs'][0]['tensors'][input_index]['shape'][1:])

    model.add(InputLayer(input_shape=shape))

def add_dense(model, operator, info, interpreter):
    bias = interpreter.tensor(operator['inputs'][-1])()
    weights = interpreter.tensor(operator['inputs'][-2])()

    layer = Dense(units=len(bias))

    model.add(layer)

    # verify channels first/last of weights
    (ls, ws) = (tuple(layer.weights[0].shape), weights.shape)
    if ls[-1]==ws[0] and ls[0]==ws[-1]:
        weights = np.transpose(weights, (1, 0))

    layer.set_weights([weights, bias])

    return layer

def add_conv2d(model, operator, data, interpreter):
    bias = interpreter.tensor(operator['inputs'][-1])()
    weights = interpreter.tensor(operator['inputs'][-2])()

    filters = weights.shape[0]
    kernel_size = (weights.shape[1], weights.shape[2])
    opt = operator['builtin_options']
    strides = (opt['stride_h'], opt['stride_w'])
    # padding = opt['padding'] # "valid" or "same"
    padding = 'valid'  # to do analize padding

    dilation_rate = (opt['dilation_h_factor'], opt['dilation_w_factor'])

    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                   padding=padding, data_format='channels_last',
                   dilation_rate=dilation_rate)

    model.add(layer)
    # verify channels first/last of weights
    (ls, ws) = (tuple(layer.weights[0].shape), weights.shape)

    if ls[-1] == ws[0] and ls[0:-1] == ws[1:]:
        weights = np.transpose(weights, (1, 2, 3, 0))

    layer.set_weights([weights, bias])
    return layer


def add_pool2d(pool_class, model, operator, data, interpreter):

    opt = operator['builtin_options']
    pool_size = (opt['filter_height'], opt['filter_width'])
    strides = (opt['stride_h'], opt['stride_w'])
    # padding = opt['padding'] # "valid" or "same"
    padding = 'valid'  # to do analize padding

    layer = pool_class(pool_size=pool_size, strides=strides,
                         padding=padding, data_format='channels_last'
                         )
    model.add(layer)

    return layer

def add_reshape(model, operator, data, interpreter):
    layer = Flatten()
    model.add(layer)
    return layer

def add_activation(func_name, model):
    layer = Activation(func_name)
    model.add(layer)
    return layer

def add_leakyrelu(model, operator, info, interpreter):
    layer = LeakyReLU(alpha=operator['builtin_options']['alpha'])
    model.add(layer)
    return layer

