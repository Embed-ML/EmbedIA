import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.keras.models import Sequential

from sklearn.datasets import make_classification
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

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

class DataGenerator:
    def __init__(self):
        self._input_data = None
        self._output_data = None
        self._test_element = None
        self._model = None
        # for classification data
        self._n_classes = 3
        self._samples_multiplier = 10

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

    @property
    def is_keras_element(self):
        return isinstance(self._test_element, tf.keras.layers.Layer)

    @property
    def is_classification_element(self):
        return isinstance(self._test_element, ClassifierMixin)
    @property
    def is_regression_element(self):
        return isinstance(self._test_element, RegressorMixin)

    def _generate_model(self):
        if self.is_keras_element:
            model = Sequential()
            if self._input_shape is not None:
                model.add(tf.keras.layers.InputLayer(input_shape=self._input_shape))
        else:
            model = DammyModel()

        model.add(self._test_element)

        self._model = model

    def _get_input_shape(self):
        # Método para generar datos de entrada aleatorios
        if self._input_shape is None:
            input_shape = self._model.input_shape
        else:
            input_shape = self._input_shape
        if input_shape[0] is None:
            input_shape = (1,) + input_shape[1:]
        return input_shape

    def _generate_inputs(self):

        if self.is_classification_element:
            self._input_data = self._generate_classification_inputs()
        else:
            self._input_data = self._generate_simple_inputs()

        return self._input_data

    def _generate_simple_inputs(self):
        input_shape = self._get_input_shape()
        self._input_data = (np.random.random(input_shape) * 3).astype('int')
        return self._input_data

    def _generate_classification_inputs(self):
        # Asumiendo que input_shape es como (None, n_features) o (batch_size, n_features)
        input_shape = self._get_input_shape()
        n_features = input_shape[1]
        n_test_samples = input_shape[0]
        n_train_samples = n_test_samples * self._samples_multiplier

        # Calcula el número apropiado de características informativas
        # Para asegurar que n_classes * n_clusters_per_class <= 2^n_informative (requerido por make_classification)
        n_clusters_per_class = 2  # Valor por defecto en make_classification
        min_n_informative = max(2, int(np.ceil(np.log2(self._n_classes * n_clusters_per_class))))
        n_informative = min(n_features, min_n_informative)

        # Datos linealmente separables con n clases
        X, y = make_classification(
            n_samples=n_train_samples+1,
            n_features=n_features,
            n_classes=self._n_classes,
            n_informative=n_informative,  # Usar el número calculado de características informativas
            n_redundant=min(n_features - n_informative, max(1, n_features // 4)),  # Características redundantes
            n_clusters_per_class=n_clusters_per_class,
            random_state=42
        )


        if hasattr(self._test_element, 'fit'):
            self._test_element.fit(X[:-1,:], y[:-1])

        self._input_data = X[-1:,:]
        return self._input_data

    def _generate_outputs(self, input_data):
        if self.is_keras_element:
            # Realizar la inferencia utilizando TensorFlow/Keras, es necesario para establecer
            # internamente las dimensiones de los pesos para luego poder cambiarlos
            output_data = self._model.predict(input_data, verbose=False)
        elif hasattr(self._test_element, 'fit_transform'):
            output_data = self._test_element.fit_transform(input_data)
        elif hasattr(self._test_element, 'predict_proba'):
            output_data = self._test_element.predict_proba(input_data)

        self._output_data = output_data

    def generate(self, test_elem, shape=None):
        self._test_element = test_elem
        self._input_shape = shape
        self._generate_model()
        self._generate_inputs()
        self._generate_outputs(self._input_data)

        return self._output_data


