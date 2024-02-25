import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential

class DammyModel:
    _model = None
    _layers = []
    _input_shape =[]

    def add(self, obj):
        self._model = obj
        self._layers = [obj]

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

    @property
    def test_element(self):
        return self._test_element
    @test_element.setter
    def test_element(self, element)->None:
        self._test_element = element
    @property
    def input_data(self):
        return self._input_data

    @property
    def output_data(self):
        return self._output_data

    def generate_inputs(self, input_shape=None):
        # Método para generar datos de entrada aleatorios
        if input_shape is None:
            input_shape = self.model.input_shape
        if input_shape[0] is None:
            input_shape = (1,) + input_shape[1:]
        self._input_data = np.random.random(input_shape)
        return self._input_data

    @abstractmethod
    def _do_generate(self, input_data):
        pass

    def generate(self, input_data=None):
        self.generate_inputs(input_data)
        self._do_generate(self._input_data)

        return self._output_data


class TFLayerDataGenerator(BaseDataGenerator):

    @property
    def model(self):
        return self._model

    @property
    def test_element(self):
        return self._test_element

    @test_element.setter
    def test_element(self, element)->None:
        self._test_element = element
        if isinstance(self._test_element, tf.keras.layers.Layer):
            self._model = Sequential()
        else:
            self._model = DammyModel()
        self._model.add(self._test_element)

    def _do_generate(self, input_data):
        layer = self._test_element
        if isinstance(layer, tf.keras.layers.Layer):
            # Realizar la inferencia utilizando TensorFlow/Keras, es necesario para establecer
            # internamente las dimensiones de los pesos para luego poder cambiarlos
            dummy_output_data = self._model.predict(input_data, verbose=False)
        elif hasattr(layer, 'fit_transform'):
            dummy_output_data = layer.fit_transform(input_data)

        self._output_data = dummy_output_data



