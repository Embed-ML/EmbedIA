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
    def frame_length(self):
        return self._target.frame_length

    @property
    def input_length(self):
        return self._target.input_length

    @property
    def sample_rate(self):
        return self._target.input_sr

    @property
    def n_frames(self):
        return self._target.n_frames

    @property
    def n_channels(self):
        if self._target.n_channels is None:
            return 1
        return self._target.n_channels

    @property
    def overlap_length(self):
        return self._target.overlap_length

    @property
    def hop_length(self):
        return self._target.hop_length

    @property
    def shape(self):
        return self._target.shape

    @property
    def window(self):
        return self._target.window

    @property
    def convert_to_db(self):
        return self._target.convert_to_db

    @property
    def top_db(self):
        return self._target.top_db

