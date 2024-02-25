import larq as lq
from tensorflow import keras
from sklearn import preprocessing
from embedia.utils import melspec

from embedia.layers.dummy_layer import DummyLayer
from embedia.layers.binary_convolution.quantconv2d import QuantConv2D
from embedia.layers.binary_dense.quantdense import QuantDense
from embedia.layers.binary_convolution.quantseparable_conv2d import QuantSeparableConv2D
from embedia.layers.convolution.separable_conv2d import SeparableConv2D
from embedia.layers.convolution.conv2d import Conv2D
from embedia.layers.convolution.depthwise_conv2d import DepthwiseConv2D
from embedia.layers.dense.dense import Dense
from embedia.layers.flatten.flatten import Flatten
from embedia.layers.pooling.pooling import Pooling
from embedia.layers.batch_normalization.batch_normalization import BatchNormalization
from embedia.layers.activation.activation import Activation
from embedia.layers.normalization.standard_scaler import StandardNormalization
from embedia.layers.normalization.min_max_scaler import MinMaxNormalization
from embedia.layers.normalization.max_abs_scaler import MaxAbsNormalization
from embedia.layers.normalization.robust_scaler import RobustNormalization
from embedia.layers.signal_processing.spectrogram import Spectrogram

dict_layers = {
    # Layers with No porpose in inference
    keras.layers.InputLayer: DummyLayer,
    keras.layers.Dropout: DummyLayer,
    keras.layers.SpatialDropout1D: DummyLayer,
    keras.layers.SpatialDropout2D: DummyLayer,
    keras.layers.SpatialDropout3D: DummyLayer,
    keras.layers.GaussianDropout: DummyLayer,
    keras.layers.AlphaDropout: DummyLayer,
    keras.layers.GaussianNoise: DummyLayer,
    keras.layers.RandomBrightness: DummyLayer,
    keras.layers.RandomContrast: DummyLayer,
    keras.layers.RandomCrop: DummyLayer,
    keras.layers.RandomFlip: DummyLayer,
    keras.layers.RandomHeight: DummyLayer,
    keras.layers.RandomRotation: DummyLayer,
    keras.layers.RandomTranslation: DummyLayer,
    keras.layers.RandomWidth: DummyLayer,
    keras.layers.RandomZoom: DummyLayer,
    lq.layers.QuantConv2D: QuantConv2D,
    lq.layers.QuantDense: QuantDense,
    lq.layers.QuantSeparableConv2D: QuantSeparableConv2D,
    keras.layers.SeparableConv2D: SeparableConv2D,
    keras.layers.DepthwiseConv2D: DepthwiseConv2D,
    keras.layers.Conv2D: Conv2D,
    keras.layers.Dense: Dense,
    keras.layers.Flatten: Flatten,
    keras.layers.BatchNormalization: BatchNormalization,
    keras.layers.Activation: Activation,
    keras.layers.ReLU: Activation,
    keras.layers.LeakyReLU: Activation,
    keras.layers.Softmax: Activation,
    # pooling layers
    keras.layers.AveragePooling1D: Pooling,  # not yet implemented in C
    keras.layers.AveragePooling2D: Pooling,
    keras.layers.AveragePooling3D: Pooling,  # not yet implemented in C
    keras.layers.MaxPooling1D: Pooling,      # not yet implemented in C
    keras.layers.MaxPooling2D: Pooling,
    keras.layers.MaxPooling3D: Pooling,      # not yet implemented in C
    # normalization layers from SKLearn
    preprocessing.StandardScaler: StandardNormalization,
    preprocessing.MinMaxScaler: MinMaxNormalization,
    preprocessing.MaxAbsScaler: MaxAbsNormalization,
    preprocessing.RobustScaler: RobustNormalization,
    # signal processing
    melspec.Melspec: Spectrogram,
}
