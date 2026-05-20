from embedia.layers.transformation.channels_adapter import ChannelsAdapter
from embedia.layers.activation.activation import Activation
from embedia.core.embedia_model import EmbediaModel
from embedia.core.layer_wrapper import OutputPredictionType
from embedia.wrappers.tensorflow.activation import TFActivationWrapper
from tensorflow.keras.layers import Activation as KerasActivation


class TensorflowModel(EmbediaModel):


    def _get_input_adapter(self):
        if self.model is None:
            return None

        #if 'Conv' not in self.model.layers[0].__class__.__name__:
        #    return None
        if hasattr(self.model.layers[0], 'data_format') and self.model.layers[0].data_format != 'channels_last':
            return None # has attribute but channel is first

        inp_shape = self.model.input_shape[1:]
        if len(inp_shape)>=2 and inp_shape[-1]>=2:
                return ChannelsAdapter(model=self, shape=inp_shape, options=self.options)

        return None

    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # add adapter if its required
        input_adapter = self._get_input_adapter()
        if input_adapter is not None:
            self._embedia_layers.append(input_adapter)

        # external normalizer to the model? => add as first layer
        self._add_processing_layers(self.options.preprocessing)

        for layer in self.model.layers: # TF/Keras layers
            obj = layer
            ly = self._create_embedia_layer(layer)
            self._embedia_layers.append(ly)

            if not isinstance(layer, KerasActivation) and hasattr(layer, 'activation') and layer.activation is not None:
                self._embedia_layers.append(Activation(self, TFActivationWrapper(layer)))

        # external post proccesing layers (ej softmax, argmax) to the model? => add as last layer
        self._add_processing_layers(self.options.postprocessing)

        self._complete_layers_shapes()
        return self._embedia_layers

    '''
    def identify_target_classes(self):
        """
        Identify the number of target classes based on the last layer of the model.

        Returns:
            int: 1 for binary classification, N for multiclass classification with N classes,
                or 0 for regression.
        """
        wrapper = self.embedia_layers[-1].wrapper
        act_fn = ''
        if wrapper.activation is not None:
            act_fn = wrapper.activation.__name__.lower()

        if act_fn == '':
            act_fn = wrapper.target.__class__.__name__.lower()

        if act_fn in ['sigmoid', 'sigmoidal', 'softsign', 'tanh']:
                return 1 # binary classification
        if act_fn == 'softmax':
            return wrapper.output_shape[-1] # multiclass classification

        return 0 # regression
        '''


def _infer_output_prediction_type(self):
    """Determine how to process the model's output based on last layer configuration.

    Returns:
        OutputPredictionType: Enum value indicating required post-processing:
            - BINARY_OUTPUT: Single sigmoid output (apply threshold)
            - CLASS_PROBABILITIES: Softmax/sigmoid outputs (apply argmax)
            - REGRESSION_OUTPUT: Single continuous output (use directly)
            - DIRECT_CLASS_ID: Raw class indices (use directly)

    Raises:
        ValueError: If output layer uses unsupported activations (tanh/softsign).
    """
    last_layer = self.embedia_layers[-1]
    wrapper = last_layer

    # Get activation function name
    act_fn = (wrapper.activation.__name__.lower() if wrapper.activation
              else wrapper.target.__class__.__name__.lower())
    output_shape = wrapper.output_shape

    # Classification cases
    if act_fn == 'sigmoid':
        return OutputPredictionType.BINARY_OUTPUT if output_shape[-1] == 1 else OutputPredictionType.CLASS_PROBABILITIES
    elif act_fn == 'softmax':
        return OutputPredictionType.CLASS_PROBABILITIES

    # Unsupported activations
    if act_fn in ['tanh', 'softsign']:
        raise ValueError(
            f"Unsupported activation '{act_fn}' in output layer. "
            "Use 'sigmoid' (binary), 'softmax' (multiclass), "
            "or no activation (regression/direct)."
        )

    # Default cases (no activation)
    return OutputPredictionType.REGRESSION_OUTPUT if output_shape[-1] == 1 else OutputPredictionType.DIRECT_CLASS_ID


