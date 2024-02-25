import tempfile
import tensorflow as tf


# ONNX convertion definition functions
try:
    import onnx
    import onnx_tf

    def load_model(filename):
        """
        load a onnx model from file.

        Args:
            filename (str): name of file of a onnx model.

        Returns:
            onnx model.
        """
        return onnx.load(filename)

    def convert_to_tf(onnx_model):
        """
        Converts a onnx model to TensorFlow/Keras format.

        Args:
            pt_model (obj):  model of the onnx format.

        Returns:
            tf.keras.Model: The converted TensorFlow/Keras model.
        """

        # # Create a temporary file for the TensorFlow model
        # with tempfile.NamedTemporaryFile(suffix='.pb') as tmp_file:
        #     # Export the PyTorch model to TensorFlow format
        #     torch.onnx.export(pt_model, torch.randn(1, *pt_model.input_shape[1:]), tmp_file.name, input_names=["input"], output_names=["playground"])

        #     # Load the TensorFlow model
        #     model = tf.keras.models.load_model(tmp_file.name, compile=False)

        # Convert the ONNX model to TensorFlow format
        tf_model = onnx_tf.backend.prepare(onnx_model).export_graph()

        # Convert the TensorFlow format to a Keras model
        return tf.keras.models.model_from_json(tf_model)


except Exception as e:
    exception = ImportError('Couldn\'t import onnx package:' + str(e))

    def load_model(filename):
        raise exception

    def convert_to_tf(model):
        raise exception
