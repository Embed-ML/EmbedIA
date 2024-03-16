import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.keras.models import Sequential


class DammyModel:
    def __init__(self):
        self._model = None
        self._layers = []
        self._input_shape = None

    def add(self, obj):
        self._layers.append(obj)

    @property
    def model(self):
        return self._model

    @property
    def layers(self):
        return self._layers
    @property
    def input_shape(self):
        return self._input_shape


class BaseDataGenerator:
    def __init__(self):
        self._input_data = None
        self._output_data = None
        self._test_element = None
        self._model = None

    @property
    def test_element(self):
        return self._test_element

    @property
    def model(self):
        return self._model

    @property
    def input_data(self):
        return self._input_data

    @property
    def output_data(self):
        return self._output_data

    def _generate_inputs(self):
        # Método para generar datos de entrada aleatorios
        if self._input_shape is None:
            input_shape = self._model.input_shape
        else:
            input_shape = self._input_shape
        if input_shape[0] is None:
            input_shape = (1,) + input_shape[1:]
        self._input_data = (np.random.random(input_shape)*3).astype('int')
        return self._input_data

    @abstractmethod
    def _do_generate(self, input_data):
        pass

    @abstractmethod
    def _generate_model(self):
        pass

    def generate(self, test_elem, shape=None):
        self._test_element = test_elem
        self._input_shape = shape
        self._generate_model()
        self._generate_inputs()
        self._do_generate(self._input_data)

        return self._output_data


class TFLayerDataGenerator(BaseDataGenerator):

    def __init__(self):
        super().__init__()

    def _generate_model(self):
        if isinstance(self._test_element, tf.keras.layers.Layer):
            model = Sequential()
        else:
            model = DammyModel()

        if self._input_shape is not None:
            model.add(tf.keras.layers.InputLayer(input_shape=self._input_shape))

        model.add(self._test_element)

        self._model = model

    def _do_generate(self, input_data):
        if isinstance(self._test_element, tf.keras.layers.Layer):
            # Realizar la inferencia utilizando TensorFlow/Keras, es necesario para establecer
            # internamente las dimensiones de los pesos para luego poder cambiarlos
            dummy_output_data = self._model.predict(input_data, verbose=False)
        elif hasattr(self._test_element, 'fit_transform'):
            dummy_output_data = self._test_element.fit_transform(input_data)

        self._output_data = dummy_output_data



