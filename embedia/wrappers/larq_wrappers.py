from embedia.core.layer_wrapper import LayerWrapper
import numpy as np

class LarqWrapper(LayerWrapper):

    @property
    def input_shape(self):
        return self._target.input_shape

    @property
    def output_shape(self):
        return self._target.output_shape

    @property
    def weights(self):
        '''
        Output format array: filters, channel, row, column
        '''
        return self._target.get_weights()[0]

    @property
    def biases(self):
        return self._target.get_weights()[1]


    def get_config(self):
        return self._target.get_config()


class LarqQuantConv2DWrapper(LarqWrapper):
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
    def strides(self):
        return self._target.strides

    @property
    def padding(self):
        if self._target.padding == 'same':
            return 1
        return 0


class LarqQuantSeparableConv2DWrapper(LarqQuantConv2DWrapper):
    @property
    def depth_weights(self):
        '''
        Output format array: filters, channel, row, column
        '''
        return self._adapt_weights(self._target.get_weights()[0])

    @property
    def point_weights(self):
        return self._adapt_weights(self._target.get_weights()[1])

    @property
    def biases(self):
        return self._target.get_weights()[2]


