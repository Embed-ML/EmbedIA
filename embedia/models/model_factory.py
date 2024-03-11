import inspect
from embedia.models import TensorflowModel, SklearnModel


class ModelFactory:
    """
    A factory class for creating instances of EmbedIA models (TensorflowModel, SklearnModel, etc.) capable of
    recognizing the architecture of the provided model_object for conversion to the C language.
    """

    @staticmethod
    def create_model(model_object, options):
        """
        Creates an instance of an EmbedIA model (TensorflowModel, SklearnModel, etc.) capable of recognizing the
        architecture of the model_object for conversion to the C language.
        Parameters:
            model_object (object): The model object to create an instance of (TensorflowModel, SklearnModel, etc.)
            options (dict): A dictionary containing options for the model.

        Returns:
            object: An instance of the appropriate model (TensorflowModel, SklearnModel, etc.) based on the
            architecture of the provided model_object.

        Raises:
            ValueError: If the provided model_object is not in the TensorFlow or scikit-learn module.
        """

        main_module = inspect.getmodule(model_object).__name__.split('.')[0]
        if main_module in ['tensorflow', 'keras']:
            return TensorflowModel(model_object, options)
        elif main_module in ['sklearn']:
            return SklearnModel(model_object, options)
        else:
            raise ValueError("Object not in TensorFlow or scikit-learn module")
