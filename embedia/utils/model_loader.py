import tensorflow as tf
from embedia.utils.converters import torch, onnx, tflite
from enum import Enum


class ModelFormat(Enum):
    """
    An enumeration of supported model formats.
    """
    TF_KERAS = 0
    PYTORCH = 1
    TFLITE = 2
    ONNX = 3
    UNKNOWN = 4


class ModelLoader:
    """
    A class that provides functions for loading and converting models.
    """

    @staticmethod
    def detect_model_format(filename):
        """
        Detects the format of a model file based on its file extension.

        Args:
            filename (str): The name of the model file.

        Returns:
            ModelFormat: An enumeration value representing the format of the
            model file.
        """
        extension = filename.split('.')[-1]

        if extension == 'h5' or extension == 'pb':
            return ModelFormat.TF_KERAS
        elif extension == 'pt':
            return ModelFormat.PYTORCH
        elif extension == 'tflite':
            return ModelFormat.TFLITE
        elif extension == 'onnx':
            return ModelFormat.ONNX
        else:
            return ModelFormat.UNKNOWN

    @staticmethod
    def convert_onnx_to_tf(onnx_model):
        """
        Converts an ONNX model to TensorFlow/Keras format.

        Args:
            onnx_model (obj): Model of the ONNX format.

        Returns:
            tf.keras.Model: The converted TensorFlow/Keras model.
        """

        # Convert the ONNX model to TensorFlow format
        return onnx.convert_to_tf(onnx_model)


    @staticmethod
    def convert_tourch_to_tf(pt_model):
        """
        Converts a PyTorch model to TensorFlow/Keras format.

        Args:
            pt_model (obj):  model of the PyTorch format.

        Returns:
            tf.keras.Model: The converted TensorFlow/Keras model.
        """

        return torch.convert_to_tf(pt_model)

    @staticmethod
    def convert_tflite_to_tf(interpreter):
        """
        Converts a TFLite model to TensorFlow/Keras format.

        Args:
            interpreter(obj): Model of the TFLite format.

        Returns:
            tf.keras.Model: The converted TensorFlow/Keras model.
        """

        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # interpreter.allocate_tensors()
        # input_shape = input_details[0]['shape']

        # # Define the TensorFlow/Keras model
        # model = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=input_shape[1:]),
        #     tf.keras.layers.Lambda(lambda x: interpreter.tensor(
        #         interpreter.get_output_details()[0]['index'])())
        # ])

        # Convierte el modelo .tflite a un modelo estándar de TensorFlow/Keras
        converter = tf.lite.TFLiteConverter.from_interpreter(interpreter)
        model = converter.convert()

        return model

    @staticmethod
    def load_model(filename):
        """
        Loads a model from a file and returns it in TensorFlow/Keras format.

        Args:
            filename (str): The name of the model file.

        Returns:
            tf.keras.Model: The loaded TensorFlow/Keras model.
        """
        # Detect the model format
        format = ModelLoader.detect_model_format(filename)

        # Convert the model to TensorFlow/Keras format
        if format == ModelFormat.TF_KERAS:
            # The model is already in TensorFlow/Keras format
            model = tf.keras.models.load_model(filename, compile=False)

        elif format == ModelFormat.PYTORCH:
            # Load the PyTorch model
            src_model = torch.load_model(filename)
            # convert to TensorFlow/Keras format
            model = torch.convert_to_tf(src_model)

        elif format == ModelFormat.TFLITE:
            # Load the TFLite model
            src_model = tflite.load_model(filename)
            # convert to TensorFlow/Keras format
            model = tflite.convert_to_tf(src_model)

        elif format == ModelFormat.ONNX:
            # Load the ONNX model
            src_model = onnx.load_model(filename)
            # convert to TensorFlow/Keras format
            model = onnx.convert_to_tf(src_model)
        else:
            raise ValueError("Unknown model format")

        return model
