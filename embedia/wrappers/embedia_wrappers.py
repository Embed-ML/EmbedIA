from embedia.core.layer_wrapper import LayerWrapper
import numpy as np

class EmbediaWrapper(LayerWrapper):

    @property
    def input_shape(self):
        return self._target.input_shape

    @property
    def output_shape(self):
        return self._target.output_shape

class EmbediaSpectrumWrapper(EmbediaWrapper):

    @property
    def n_fft(self):
        return self._target.n_fft

    @property
    def n_mels(self):
        return self._target.n_mels

    @property
    def input_length(self):
        return self._target.input_length

    @property
    def sample_rate(self):
        return self._target.input_fs

    @property
    def n_blocks(self):
        return self._target.n_blocks

    @property
    def noverlap(self):
        return self._target.noverlap

    @property
    def step(self):
        return self._target.step

    @property
    def shape(self):
        return self._target.shape
