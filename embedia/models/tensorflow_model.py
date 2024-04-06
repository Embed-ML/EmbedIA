from embedia.layers.transformation.channels_adapter import ChannelsAdapter
from embedia.layers.activation.activation import Activation
from embedia.core.embedia_model import EmbediaModel
from embedia.wrappers.tensorflow_wrappers import TFActivationWrapper
from tensorflow.keras.layers import Activation as KerasActivation


class TensorflowModel(EmbediaModel):


    def _get_input_adapter(self):
        if self.model is None:
            return None

        if hasattr(self.model.layers[0], 'data_format') and self.model.layers[0].data_format != 'channels_last':
            return None # has attribute but channel is first

        inp_shape = self.model.input_shape[1:]
        if len(inp_shape)>=3 and inp_shape[-1]>=2:
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
        if self.options.normalizer is not None:
            obj = self.options.normalizer
            ly = self._create_embedia_layer(obj)
            self._embedia_layers.append(ly)

        for layer in self.model.layers: # TF/Keras layers
            obj = layer
            ly = self._create_embedia_layer(layer)
            self._embedia_layers.append(ly)

            if not isinstance(layer, KerasActivation) and hasattr(layer, 'activation') and layer.activation is not None:
                self._embedia_layers.append(Activation(self, TFActivationWrapper(layer)))


        self._complete_layers_shapes()
        return self._embedia_layers
