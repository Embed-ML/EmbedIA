from embedia.core.embedia_model import EmbediaModel


class SklearnModel(EmbediaModel):

    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # external normalizer to the model? => add as first layer
        if self.options.normalizer is not None:
            obj = self.options.normalizer
            ly = self._create_embedia_layer(obj)
            self._embedia_layers.append(ly)

        ly = self._create_embedia_layer(self.model)
        self._embedia_layers.append(ly)

        self._complete_layers_shapes()

        return self.embedia_layers

