from embedia.core.layer_wrapper import LayerWrapper
from tensorflow.keras.layers import Activation
import numpy as np
import re

class TensorflowWrapper(LayerWrapper):

    @property
    def input_shape(self):
        return self._target.input_shape

    @property
    def output_shape(self):
        return self._target.output_shape

    @property
    def name(self):
        return self._target.name

    @property
    def activation(self):
        if hasattr(self._target, 'activation'):
            return self._target.activation
        return None

    # @property
    # def weights(self):
    #     return self._target.get_weights()[0]
    #
    # @property
    # def biases(self):
    #     return self._target.get_weights()[1]
    #

class TFDenseWrapper(TensorflowWrapper):

    @property
    def weights(self):
        return self._target.get_weights()[0]

    @property
    def biases(self):
        return self._target.get_weights()[1]



class TFPoolWrapper(TensorflowWrapper):

    @property
    def strides(self):
        return self._target.strides

    @property
    def pool_size(self):
        return self._target.pool_size

    @property
    def dimensions(self):
        return len(self._target.pool_size)

    @property
    def function_name(self):
        return self._target.pool_function.__name__.lower()[0:3]


class TFPaddingWrapper(TensorflowWrapper):

    @property
    def padding(self):
        return self._target.padding


class TFConv2DWrapper(TensorflowWrapper):

    def _adapt_weights(self, weights):
        '''
         Input fromat array: row, col, channel, filters
         Output format array: filters, channel, row, column
        '''
        _row, _col, _chn, _filt = weights.shape
        arr = np.zeros((_filt, _chn, _row, _col))
        for row, elem in enumerate(weights):
            for column, elem2 in enumerate(elem):
                for channel, elem3 in enumerate(elem2):
                    for filters, value in enumerate(elem3):
                        arr[filters, channel, row, column] = value
        return arr

    @property
    def weights(self):
        '''
        Output format array: filters, channel, row, column
        '''
        return self._adapt_weights(self._target.get_weights()[0])

    @property
    def biases(self):
        return self._target.get_weights()[1]

    @property
    def strides(self):
        return self._target.strides

    @property
    def padding(self):
        if self._target.padding == 'same':
            return 1
        return 0

class TFSeparableConv2DWrapper(TFConv2DWrapper):
    @property
    def depth_weights(self):
        '''
        Output format array: filters, channel, row, column
        '''
        return self._adapt_weights(self._target.get_weights()[0])

    @property
    def point_weights(self):
        '''
        Output format array: filters, channel, row, column
        '''
        return self._adapt_weights(self._target.get_weights()[1])

    @property
    def biases(self):
        return self._target.get_weights()[2]


class TFBatchNormWrapper(TensorflowWrapper):

    @property
    def gamma(self):
        return self._target.get_weights()[0]

    @property
    def beta(self):
        return self._target.get_weights()[1]

    @property
    def moving_mean(self):
        return self._target.get_weights()[2]

    @property
    def moving_variance(self):
        return self._target.get_weights()[3]

    @property
    def epsilon(self):
        return self._target.epsilon


class TFActivationWrapper(TensorflowWrapper):

    @property
    def function_name(self):
        '''
        This method must provide a string with an activation function name. This name must be lower case
        and composed of the name funcition without spaces, underscore, etc. For example, Leaky ReLU activation name
        is 'leakyrelu'. Some others examples: 'linear', 'relu', 'leakyrelu', 'softplus', 'softmax', 'tanh', 'sigmoid'

        Must be taken in account that the target object can be a Tensorflow Activation object or another object with
        the activation property.
        '''
        if not hasattr(self._target, 'activation'):
            name = self._target.__class__.__name__ # target is a class from keras.layers as ReLU, LeakyReLU or Softmax class
        elif hasattr(self._target.activation, '__name__'): # target is a class like Dense with activation as function
            name = self._target.activation.__name__  # activation is a function
        else:
            name = self._target.activation.__class__.__name__ #  target is a class like Dense with activation as object

        return re.sub(r'_[^_]*$', '', name.lower())  # delete text after underscore. Ej: softmax_v2 => softmax

    @property
    def leakyrelu_alpha(self):
        if hasattr(self._target, 'activation'):
            return self._target.activation.alpha  # target.activation is an object with alpha property
        return self._target.alpha  # target is a keras.layers.LeakyReLU with alpha property
