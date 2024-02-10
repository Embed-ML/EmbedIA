import numpy as np
import matplotlib.pyplot as plt

class Melspec(object):
	'''
		Class that processes an audio and results in a 
		specrogram according to the class parameters it has.
	'''
	def __init__(self, n_fft, n_mels, noverlap = None):
		'''
			Class constructor.

			Parameters
			----------
			n_fft: 
				The number of data points used in each block for the DFT.
			n_mels: 
				Number of mel bins.
			noverlap: 
				The number of points of overlap between blocks. 
		'''
		# super().__init__(**kwargs)

		self._class         = type(self)
		self._class.name_   = 'spectrogram'
		# self.name_ 			= 'spectrogram'
		self.n_fft 			= n_fft
		self.n_mels 		= n_mels
		self.noverlap 		= noverlap
		self.spec 			= None
		self.shape 			= None
		self.step 			= None
		self.n_blocks 		= None
		self.input_length 	= None
		self.input_fs 		= None
		self.input_shape 	= None #(self.input_length,)
		self.output_shape 	= None #(self.n_blocks,self.n_mels,1)

	def set_params(self, n_fft = None, n_mels = None, noverlap = None, spec = None):
		'''
			Function to update parameters.

			Parameters
			----------
			n_fft: 
				The number of data points used in each block for the DFT.
			n_mels: 
				Number of mel bins.
			noverlap: 
				The number of points of overlap between blocks.
			spec: 
				Vector containing a spectrogram.
		'''
		if not n_fft is None:
			self.n_fft = n_fft
		if not n_mels is None:
			self.n_fft = n_mels
		if not noverlap is None:
			self.noverlap = noverlap
		if not spec is None:
			self.spec = spec
		self.input_shape = (self.input_length,)
		self.output_shape = (self.n_blocks,self.n_mels,1)

	def reset(self):
		'''
			Function that resets the values 
			of the spectrogram calculation results 
		'''
		self.spec = None
		self.shape = None
		self.step = None
		self.n_blocks = None
		self.input_length = None
		self.input_fs = None

	def ready(self):
		'''
			Function that indicates if the operation 
			has been carried out or data is missing.

			Return
			------
				True: if all data is ok 
				False: if data is missing
		'''
		if self.step is None:
			return False
		elif self.n_blocks is None:
			return False
		elif self.input_length is None:
			return False
		elif self.input_fs is None:
			return False
		elif self.shape is None:
			return False
		else:
			return True

	def report(self):
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
		info += f"\tn_fft = {self.n_fft}\n"
		info += f"\tn_mels = {self.n_mels}\n"
		info += f"\tnoverlap = {self.noverlap}\n"
		
		info += "Spectrogram - Results\n"
		if self.ready():
			info += f"\tSpec shape: {self.shape}\n"
			info += f"\tstep = {self.step}\n"
			info += f"\tn_blocks = {self.n_blocks}\n"
			info += f"\tinput_length = {self.input_length}\n"
			info += f"\tinput_fs = {self.input_fs}\n"
		if self.spec is None:
			info+="\tWARNING: Signal not yet processed\n"

		print(info)
		return info

	def process_melspectrogram(self, signal, fs, report = False):
		'''
			Function processes the vector in signal using 
			the parameters of the instance for its configuration.
			
			Parameters
			----------
				signal:
					Original time series.
				fs:
					Signal sampling rate.
				report:
					Indicate with True if the report should be displayed at the end and False otherwise.

			Returns
			-------
			spec: 
				Matrix resulting from the calculation of the spectrogram.
		'''
		if self.noverlap is None:
			self.noverlap = 0
		self.noverlap = int(self.noverlap)

		self.step = self.n_fft-self.noverlap
		self.input_length = len(signal)
		self.input_fs = fs

		starts  = np.arange(0,len(signal),self.step,dtype=int)
		starts  = starts[starts + self.step < len(signal)]
		len_nfft_nmels = (self.n_fft//2)//self.n_mels
		xns = []

		self.n_blocks = len(starts)
		
		for start in starts:        
			ts_window = np.abs(np.fft.fft(signal[start:start+self.n_fft]))
			ts_window_n_mel = []
			for i in range(self.n_mels):
				mean_value = ts_window[i*len_nfft_nmels:i*len_nfft_nmels+len_nfft_nmels].mean()
				ts_window_n_mel.append(mean_value)
			xns.append(ts_window_n_mel)

		self.spec = np.array(xns)
		self.shape = self.spec.shape
		assert self.spec.shape[0] == self.n_blocks

		self.input_shape = (self.input_length,)
		self.output_shape = (self.n_blocks,self.n_mels,1)

		if report:
			self.report()
			
		return self.spec

	def calculate_params(self, len_signal, fs, report = False):
		'''
			Function that calculates the execution 
			parameters of melspectrogram.
			
			Parameters
			----------
				len_signal:
					Length of original time series.
				fs:
					Signal sampling rate.
				report:
					Indicate with True if the report should be displayed at the end and False otherwise.

			Returns
			-------
			spec: 
				Matrix resulting from the calculation of the spectrogram.
		'''
		if self.noverlap is None:
			self.noverlap = 0
		self.noverlap = int(self.noverlap)

		self.step = self.n_fft-self.noverlap
		self.input_length = len_signal
		self.input_fs = fs

		starts  = np.arange(0,self.input_length,self.step,dtype=int)
		starts  = starts[starts + self.step < self.input_length]
		len_nfft_nmels = (self.n_fft//2)//self.n_mels

		self.n_blocks = len(starts)
		
		self.shape = (self.n_blocks, self.n_mels)

		if report:
			self.report()

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
		assert self.ready(), "You must execute the function process_melspectrogram() before graphing."

		if not title is None:
			plt.title(title)

		plt.xlabel("Time")
		plt.ylabel("Frecuency")
		if grayscale:
			plt.imshow(self.spec.T, cmap="gray")
		else:
			plt.imshow(self.spec.T)
		plt.show()
