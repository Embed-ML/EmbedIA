import numpy as np
from embedia.native.signals.spectrum_base import SpectrogramBase


class STFT(SpectrogramBase):
	'''
    Class that implements Short-Time Fourier Transform (STFT) processing.
    '''

	def __init__(self, n_fft, overlap_length = 0, n_channels = 1, window_type='hann', input_length=22100, input_fs=22100, padding=False, convert_to_db=False):

		'''
        STFT constructor.

        Parameters
        ----------
        n_fft : int
            The number of data points used in each block for the DFT.
        overlap_length : int, optional
            The number of points of overlap between blocks.
        window_type : str or tuple or array_like
            Desired window to use. Default is 'hann'.
        '''
		super().__init__(n_fft, overlap_length, n_channels, window_type, input_length, input_fs, padding, convert_to_db)

	def compute_spectrum(self, frame):
		frame = frame.astype(np.float32)
		spectrum = np.fft.fft(frame)
		return np.abs(spectrum[:len(spectrum) // 2])