from embedia.core.layer import Layer
from embedia.model_generator.project_options import ModelDataType
from math import log2

class KnnBaseLayer(Layer):
    """
    Develop info:
    This class defines the basic behavior of an EmdedIA layer/element. It defines
    methods/properties to obtain information related to the inputs and outputs of
    the layer/element such as its shape, number of elements, EmbedIA associated
    data type.
    It also implements methods to generate the C code necessary for debugging
    function and invocation of the C function associated to the layer/element.
    This function must be implemented in some .c file its prototype declared
    in respective .h. The name of function can be anything but an EmbedIA naming
    rule is recomended: LayerClassName+"_layer". Example for Conv2D class should
    be named conv2d_layer.
    The invoke function receives an input and an output parameter with the parameter's
    name that are used in the predict function of the model.
      - model = None # EmbedIA model
      - target = None # object for data access
      - support_quantization = False # support quantized data. Default False
      - inplace_output = False # process output in input variable
      - use_data_structure = False # requires data structure definition an initialization
    """

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
        return super().required_files + [('knn.h', 'knn.c'), ('distances.h', 'distances.c')]


    def calculate_params(self):
        """
         calculates trainable and non trainable parameters of layer
         Returns

         -------
         int
             tuple (#trainable params, # non trainable params)

         """
        n_samples = self._wrapper.n_samples
        n_features = self._wrapper.n_features
        #n_neighbors = self._wrapper.n_neighbors

        trainable = 0
        non_trainable = n_samples * n_features + n_samples

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        calculates amount of multiplication and accumulation operations
        Returns
        -------
        int
            amount of multiplication and accumulation operations

        """
        n_samples = self._wrapper.n_samples
        n_features = self._wrapper.n_features
        n_neighbors = self._wrapper.n_neighbors
        MACs = (n_samples * n_features + n_samples * int(log2(n_samples)) + n_neighbors)

        return MACs


    def calculate_memory(self):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """
        n_samples = self._wrapper.n_samples
        n_features = self._wrapper.n_features
        n_neighbors = self._wrapper.n_neighbors

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)
        dt_size /= 8
        mem_size = (n_samples * n_features * dt_size) + (n_samples * dt_size);

        return mem_size



