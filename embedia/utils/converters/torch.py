import tempfile
import tensorflow as tf


# Pytorch convertion definition functions
try:
    import torch

    def load_model(filename):
        """
        load a PyTorch model from file.

        Args:
            filename (str): name of file of a PyTorch model.

        Returns:
            PyTorch model.
        """
        return torch.load(filename)

    def convert_to_tf(pt_model):
        """
        Converts a PyTorch model to TensorFlow/Keras format.

        Args:
            pt_model (obj):  model of the PyTorch format.

        Returns:
            tf.keras.Model: The converted TensorFlow/Keras model.
        """

        # Create a temporary file for the TensorFlow model
        with tempfile.NamedTemporaryFile(suffix='.pb') as tmp_file:
            # Export the PyTorch model to TensorFlow format
            torch.onnx.export(pt_model, torch.randn(1, *pt_model.input_shape[1:]), tmp_file.name, input_names=["input"], output_names=["playground"])

            # Load the TensorFlow model
            model = tf.keras.models.load_model(tmp_file.name, compile=False)

        return model

except Exception as e:
    exception = ImportError('Couldn\'t import torch package:' + str(e))

    def load_model(filename):
        raise exception

    def convert_to_tf(model):
        raise exception
