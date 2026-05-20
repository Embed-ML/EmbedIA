import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.signal.windows import get_window

class SpectrogramBase(object):
    """
    Base class for spectrogram computation from audio signals.
    Supports configurable frame/window parameters, overlap, and multi-channel processing.

    Parameters
    ----------
    frame_length : int
        Length of each analysis window (in samples). Also known as `win_length` or `n_fft`.
    overlap_length : int, optional (default=0)
        Number of overlapping samples between consecutive frames. Must satisfy `0 <= overlap_length < frame_length`.
    n_channels : int, optional (default=1)
        Number of input audio channels (1=mono, 2=stereo). If >1, outputs will have a channel dimension.
    window_type : str, optional (default='hann')
        Windowing function name (e.g., 'hann', 'hamming', 'blackman'). Must be compatible with `scipy.signal.get_window`.
    input_length : int, optional (default=22100)
        Expected length of input signals (in samples). Used to precompute buffer sizes.
    input_sr : int, optional (default=22100)
        Sampling rate of input signals (in Hz). Critical for frequency-axis computations.
    padding : bool, optional (default=False)
        If True, zero-pads the input to include partial frames at the end. If False, discards trailing samples.
    convert_to_db : bool, optional
            If True, converts output to decibel scale

    Attributes
    ----------
    hop_length : int
        Number of samples between consecutive frame starts (`frame_length - overlap_length`).
    n_frames : int
        Total number of time frames in the output spectrogram.
    output_shape : tuple
        Expected shape of the spectrogram output (depends on `n_channels` and `n_frames`).

    Notes
    -----
    - For compatibility with Librosa: Set `overlap_length = frame_length // 4` (25% overlap).
    - For compatibility with SciPy: Use `padding=False` to match `scipy.signal.stft` behavior.
    """
    def __init__(self, frame_length, overlap_length=0, n_channels=1, window_type='hann',
                 input_length=22100, input_sr=22100, padding=False, convert_to_db=False, top_db=80.0):

        # Initialize all attributes to None to prevent inconsistent state
        # before parameters are validated and processed in `set_params`.

        self._class = type(self)
        # Aliases for potential metaprogramming or naming consistency
        self._class.name = self._class._name = self._class.name_ = 'stft'

        self.EPS = 1e-8  # Small constant for numerical stability (e.g., log scaling)

        # Frame and window parameters
        self.frame_length   = None  # Analysis window length in samples
        self.overlap_length = None  # Number of overlapping samples between frames
        self.hop_length     = None  # Step size between frames: frame_length - overlap_length
        self.window_type    = None  # Type of window function (e.g., 'hann', 'hamming')

        # Input signal properties
        self.input_length   = None  # Expected length of input signal in samples
        self.input_sr       = None  # Sampling rate of input signal (Hz)
        self.input_shape    = None  # Shape of input tensor: [n_channels, input_length]

        # Output spectrogram properties
        self.n_frames       = None  # Number of time frames in the output
        self.output_shape   = None  # Shape of output spectrogram: [n_channels, n_freq_bins, n_frames]
        self.shape          = None  # Common alias for output_shape (can be a property)

        # Internal state (optional)
        self.spectrum       = None  # Last computed spectrogram (used in derived classes)
        self.padding        = None  # Whether to zero-pad to include final partial frame
        self.convert_to_db  = None  # Convert spectrogram to db (amplitude_to_db)
        self.top_db         = None  # Maximum decibel value for amplitude scaling (like librosa)

        # === Centralized parameter setup ===
        # All parameters are validated and assigned here to ensure consistency
        # and avoid partial or invalid configuration.
        self.set_params(
            frame_len=frame_length,
            overlap_len=overlap_length,
            n_channels=n_channels,
            window_type=window_type,
            input_len=input_length,
            input_sr=input_sr,
            padding=padding,
            convert_to_db=convert_to_db,
            top_db = top_db
        )


    @abstractmethod
    def compute_spectrum(self, frame):
        """Calculate spectrum for a frame (must implement subclass)"""
        pass

    def set_params(self, frame_len=None, overlap_len=None, n_channels=None, window_type=None,
                   input_len=None, input_sr=None, padding=None, spec=None, convert_to_db=None, top_db=None):
        '''
        Updates STFT parameters and computes derived properties (e.g., n_blocks).
        Validates critical constraints (e.g., n_frames > n_overlap).

        Parameters
        ----------
        frame_len : int, optional
            Number of points per FFT window. Must be > n_overlap.
        overlap_len : int, optional
            Number of overlap points between windows. Must be < n_fft.
        n_channels : int, optional
            Number of input audio channels (1 for mono, 2+ for multi-channel).
        window_type : str, optional
            Window function name (e.g., 'hann', 'blackman'), see scipy.signal.windows.get_window().
        input_len : int, optional
            Length of input signal in samples. Must be >= n_fft if padding=False.
        input_sr : int, optional
            Input sample rate in Hz.
        padding : bool, optional (default=False)
            If True, zero-pads the end to include partial frames.
        spec : np.ndarray, optional
            Precomputed spectrogram (overrides other params if provided).
        convert_to_db : bool, optional
            If True, converts output to decibel scale

        Raises
        ------
        ValueError
            If n_fft <= n_overlap, input_length < n_fft (without padding), or other invalid combinations.
        '''
        # Update basic params
        if frame_len is not None:
            self.frame_length = frame_len
        if overlap_len is not None:
            self.overlap_length = overlap_len
        self.n_channels = n_channels
        if input_len is not None:
            self.input_length = input_len
        if input_sr is not None:
            self.input_sr = input_sr
        if window_type is not None:
            self.window_type = window_type
            self.window = get_window(window_type, self.frame_length, fftbins=False).astype(np.float32)

        if spec is not None:
            self.spectrum = spec

        # --- Input Validation ---
        if not self.frame_length is None and not self.overlap_length is None:
            if self.frame_length <= self.overlap_length:
                raise ValueError(f"frame_length ({self.frame_length}) must be > overlap_length ({self.overlap_length})")

        if not self.frame_length is None and not self.input_length is None and not padding:
            if self.input_length < self.frame_length:
                raise ValueError(
                    f"input_length ({self.input_length}) < frame_length ({self.frame_length}). "
                    "Use padding=True for short signals."
                )

        # --- Derived Properties ---
        self.hop_length = self.frame_length - self.overlap_length
        # Set input shape (mono vs multi-channel)
        if self.n_channels is None:
            self.input_shape = (self.input_length,)
        else:
            self.input_shape = (self.n_channels, self.input_length)

        if not padding is None:
            self.padding = padding

        if not convert_to_db is None:
            self.convert_to_db = convert_to_db

        if not top_db is None:
            self.top_db = top_db

        # Compute n_blocks
        if not self.frame_length is None and not self.input_length is None and not self.overlap_length is None:
            if self.padding:
                # Librosa-style: Include partial frames via padding
                # self.n_frames = 1 + (self.input_length - self.frame_length) // self.hop_length
                self.n_frames = self.input_length//self.hop_length
            else:
                # Standard: Discard non-full windows
                self.n_frames = (self.input_length - self.overlap_length) // self.hop_length
            self.n_frames = max(1, self.n_frames)  # Ensure at least 1 block

        # Set output shape (adds singleton dim for compatibility)
        if self.n_channels is None:
            self.output_shape = (self.n_frames, self.frame_length // 2)
        else:
            self.output_shape = (self.n_channels, self.n_frames, self.frame_length // 2)

        self.shape = self.output_shape


    def reset(self):
        '''
            Function that resets the values
            of the spectrogram calculation results
        '''
        self.spectrum = None
        self.shape = None
        self.hop_length = None
        self.n_frames = None
        self.input_length = None
        self.input_sr = None

    def ready(self):
        '''
            Function that indicates if the operation
            has been carried out or data is missing.

            Return
            ------
                True: if all data is ok
                False: if data is missing
        '''
        required_attrs = ['hop_length', 'n_frames', 'input_length', 'input_sr', 'shape']
        return all(getattr(self, attr) is not None for attr in required_attrs)

    def report(self, return_info = False):
        '''
            Function that prints a report of
            parameters and latest results.

            Return
            ------
            info: string
                Printed information.
        '''
        info = ""
        info += "Spectrogram - Parameters\n"
        info += f"\tn_frames = {self.n_frames}\n"
        #info += f"\tn_mels = {self.n_mels}\n"
        info += f"\toverlap_length = {self.overlap_length}\n"

        info += "Spectrogram - Results\n"
        if self.ready():
            info += f"\tSpec shape: {self.shape}\n"
            info += f"\thop_length = {self.hop_length}\n"
            info += f"\tn_frames = {self.n_frames}\n"
            info += f"\tinput_length = {self.input_length}\n"
            info += f"\tinput_sr = {self.input_sr}\n"
        if self.spectrum is None:
            info+="\tWARNING: Signal not yet processed\n"

        print(info)
        if return_info:
            return info

        return None

    def process(self, signal, fs, report=False):
        '''
        Processes single or multi-channel signal using STFT.

        Parameters
        ----------
        signal : np.ndarray
            Input signal, can be:
            - 1D array (single channel) shape: (samples,)
            - 2D array (multi-channel) shape: (channels, samples)
        fs : int
            Sampling rate in Hz
        report : bool, optional
            If True, prints processing report

        Returns
        -------
        np.ndarray
            Spectrogram with shape:
            - (frames, bins) for single channel input
            - (channels, frames, bins) for multi-channel input
        '''

        # Ensure signal is at least 2D (channels × samples)
        if len(signal.shape) == 1 or (len(signal.shape) == 2 and signal.shape[0] == 1):
            if len(signal.shape) == 2:
                signal = signal[0]
            signal = np.atleast_2d(signal)
            channels = None # 1D input & 2D output
            signal = signal-signal.mean()
        else:
            channels = signal.shape[0] # 2D input & 3D output
            signal -= signal.mean(axis=1, keepdims=True)
        

        self.set_params(n_channels=channels, input_len=signal.shape[1], input_sr=fs)
        #self.n_channels = signal.shape[0]  # Update channels attribute
        #self.input_length = signal.shape[1]
        #self.input_sr = fs

        # Calculate processing parameters
        # self.step = self.n_fft - self.n_overlap

        if not self.padding:
            starts = np.arange(0, self.input_length - self.frame_length + 1, self.hop_length, dtype=int)
            # self.n_blocks = len(starts)
        else: 
            starts  = np.arange(0,self.input_length,self.hop_length,dtype=int)
            starts  = starts[starts + self.hop_length < self.input_length]

        # Initialize output array
        # output_shape = (self.n_channels, self.n_blocks, self.n_fft // 2)
        if len(self.output_shape) == 2:
            spectrograms = np.zeros((1, *self.output_shape))
        else:
            spectrograms = np.zeros(self.output_shape)

        # Process each channel
        for ch in range(spectrograms.shape[0]):
            channel_spec = []
            for start in starts:
                # Extract frame
                if not self.padding: 
                    frame = signal[ch, start:start + self.frame_length]
                else: 
                    frame = np.zeros((self.frame_length,))
                    frame[:signal[ch, start:start + self.frame_length].shape[0]] = signal[ch, start:start + self.frame_length]

                # Apply window and transform
                frame_spec = self.compute_spectrum(frame * self.window)

                if self.convert_to_db:
                    frame_spec = 20 * np.log10(np.maximum(frame_spec, self.EPS))
                    if self.top_db is not None:
                        peak = np.max(frame_spec)
                        frame_spec = np.maximum(frame_spec, peak - self.top_db)
                
                channel_spec.append(frame_spec)

            spectrograms[ch] = np.array(channel_spec)

        # Update class attributes
        if self.n_channels is None: # 1D input & 2D output
            spectrograms = spectrograms.squeeze()  # Remove single-dimensional entries

        self.spectrum = spectrograms
        self.shape = self.spectrum.shape

        # Set output shapes
        #self.input_shape = signal.shape
        #if self.n_channels == 1:
        #	self.output_shape = (self.n_blocks, self.n_fft // 2)
        #else:
        #	self.output_shape = (self.n_channels, self.n_blocks, self.n_fft // 2)

        if report:
            self.report()

        return self.spectrum


    def plot(self,title = None, grayscale = False):
        '''
            The function graphs the image resulting from the calculation
            of the last operation performed, if it has not been processed
            yet it will result in an error.

            Parameters
            ----------
            title:
                Title to add to the image.
            grayscale: boolean
                Indicates if grayscale should be used.
        '''
        assert self.ready(), "You must execute the function process() before graphing."

        if not title is None:
            plt.title(title)

        plt.xlabel("Time")
        plt.ylabel("Frecuency")
        if grayscale:
            plt.imshow(self.spectrum.T, cmap="gray")
        else:
            plt.imshow(self.spectrum.T)
        plt.show()
