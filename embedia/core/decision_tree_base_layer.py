from embedia.core.layer import Layer, EmbediaFile

class DecisionTreeBaseLayer(Layer):

    def __init__(self, model, wrapper, **kwargs):
        """
        Constructor that receives:
            - EmbedIA Model
            - object (Keras, SkLearn, etc.) associated to the EmbedIA layer
        Parameters
        ----------
        wrapper : object
            layer/object is associated to this EmbedIA layer/element. For
            example, it can receive a Keras layer or a SkLearn scaler.
        Returns
        -------
        None.
        """
        super().__init__(model, wrapper, **kwargs)

    @property
    def required_files(self):
        '''
        retorna una lista de tuplas indicando los nombres de los archivos donde se encuentra la definicion de
        tipos de datos (.h) y la implementación de las funciones (.c) requeridos por la capa/elemento
        '''
        return super().required_files + [
            (EmbediaFile('decision_tree.h',
                         {'DTREE_MAX_NODES': self.wrapper.node_count,
                                 'DTREE_N_FEATURES': self.wrapper.n_features,
                                 'DTREE_N_CLASSES': self.wrapper.n_classes}
                         ),
             EmbediaFile('decision_tree.c'))
        ]