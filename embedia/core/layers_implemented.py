import larq as lq
from tensorflow import keras
from sklearn import preprocessing
from embedia.utils import melspec

from embedia.core.dummy_layer import DummyLayer
from embedia.layers.binary_convolution.quantconv2d import QuantConv2D
from embedia.layers.binary_dense.quantdense import QuantDense
from embedia.layers.binary_convolution.quantseparable_conv2d import QuantSeparableConv2D
from embedia.layers.convolution.separable_conv2d import SeparableConv2D
from embedia.layers.convolution.conv2d import Conv2D
from embedia.layers.convolution.depthwise_conv2d import DepthwiseConv2D
from embedia.layers.dense.dense import Dense
from embedia.layers.reshaping.flatten import Flatten
from embedia.layers.pooling.pooling import Pooling
from embedia.layers.batch_normalization.batch_normalization import BatchNormalization
from embedia.layers.activation.activation import Activation
from embedia.layers.reshaping.zero_padding2d import ZeroPadding2D
from embedia.layers.normalization.standard_scaler import StandardNormalization
from embedia.layers.normalization.min_max_scaler import MinMaxNormalization
from embedia.layers.normalization.max_abs_scaler import MaxAbsNormalization
from embedia.layers.normalization.robust_scaler import RobustNormalization
from embedia.layers.signal_processing.spectrogram import Spectrogram
from embedia.layers.normalization.standard_scaler import Normalization

from embedia.wrappers.tensorflow_wrappers import (
    TensorflowWrapper,
    TFActivationWrapper,
    TFBatchNormWrapper,
    TFConv2DWrapper,
    TFDenseWrapper,
    TFPaddingWrapper,
    TFPoolWrapper,
    TFSeparableConv2DWrapper
)

from embedia.wrappers.sklearn_wrappers import (
    SKLMaxAbsScalerWrapper,
    SKLMinMaxScalerWrapper,
    SKLStandardScalerWrapper,
    SKLRobustScalerWrapper
)

dict_layers = {
    # Layers with No porpose in inference
    keras.layers.InputLayer: (DummyLayer, TensorflowWrapper),
    keras.layers.Dropout: (DummyLayer, TensorflowWrapper),
    keras.layers.SpatialDropout1D: (DummyLayer, TensorflowWrapper),
    keras.layers.SpatialDropout2D: (DummyLayer, TensorflowWrapper),
    keras.layers.SpatialDropout3D: (DummyLayer, TensorflowWrapper),
    keras.layers.GaussianDropout: (DummyLayer, TensorflowWrapper),
    keras.layers.AlphaDropout: (DummyLayer, TensorflowWrapper),
    keras.layers.GaussianNoise: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomBrightness: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomContrast: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomCrop: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomFlip: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomHeight: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomRotation: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomTranslation: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomWidth: (DummyLayer, TensorflowWrapper),
    keras.layers.RandomZoom: (DummyLayer, TensorflowWrapper),
    lq.layers.QuantConv2D: (QuantConv2D, None),
    lq.layers.QuantDense: (QuantDense, None),
    lq.layers.QuantSeparableConv2D: (QuantSeparableConv2D, None),
    keras.layers.SeparableConv2D: (SeparableConv2D, TFSeparableConv2DWrapper),
    keras.layers.DepthwiseConv2D: (DepthwiseConv2D, TFConv2DWrapper),
    keras.layers.Conv2D: (Conv2D, TFConv2DWrapper),
    keras.layers.Dense: (Dense, TFDenseWrapper),
    keras.layers.Flatten: (Flatten, TensorflowWrapper),
    keras.layers.BatchNormalization: (BatchNormalization, TFBatchNormWrapper),
    keras.layers.Activation: (Activation, TFActivationWrapper),
    keras.layers.ReLU: (Activation, TFActivationWrapper),
    keras.layers.LeakyReLU: (Activation, TFActivationWrapper),
    keras.layers.Softmax: (Activation, TFActivationWrapper),
    # pooling layers
    keras.layers.AveragePooling1D: (Pooling, None),   # not yet implemented in C
    keras.layers.AveragePooling2D: (Pooling, TFPoolWrapper),
    keras.layers.AveragePooling3D: (Pooling, None),   # not yet implemented in C
    keras.layers.MaxPooling1D: (Pooling, None),       # not yet implemented in C
    keras.layers.MaxPooling2D: (Pooling, TFPoolWrapper),
    keras.layers.MaxPooling3D: (Pooling, None),       # not yet implemented in C
    keras.layers.ZeroPadding2D: (ZeroPadding2D, TFPaddingWrapper),
    # normalization layers from SKLearn
    preprocessing.StandardScaler: (Normalization, SKLStandardScalerWrapper),
    preprocessing.MinMaxScaler: (Normalization, SKLMinMaxScalerWrapper),
    preprocessing.MaxAbsScaler: (Normalization, SKLMaxAbsScalerWrapper),
    preprocessing.RobustScaler: (Normalization, SKLRobustScalerWrapper),
    # signal processing
    melspec.Melspec: (Spectrogram, None)
}
