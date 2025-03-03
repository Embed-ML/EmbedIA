from embedia.core.knn_base_layer import KnnBaseLayer
import numpy as np


class KNeighborsClassifier(KnnBaseLayer):
    support_quantization = False  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    @property
    def function_implementation(self):
        """
        Generate C code with the initialization function of the additional
        structure (defined in "knn.h") required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """
        name = self.name
        struct_type = self.struct_data_type
        (data_type, data_converter) = self.model.get_type_converter()

        data_fit = self._wrapper.fit_x

        (conv_data_fit, quant_params) = self.convert_to_embedia_data(data_converter, data_fit)

        init_knn_layer = f'''
        
    {struct_type} init_{name}_data(void){{
    
        uint16_t n_neighbors = {self._wrapper.n_neighbors};
        uint32_t n_samples = {self._wrapper.n_samples};
        uint16_t n_features = {self._wrapper.n_features};
        uint16_t n_classes = {self._wrapper.n_classes};

    '''

        features_data = "\n".join(
            f"/* {cls} */ " + ", ".join(map(str, row)) + "," for cls, row in zip(self._wrapper.y, conv_data_fit))
        features_data = "{\n" + features_data + "\n}"

        ids_data = ','.join([str(y) for y in self._wrapper.y])
        ids_data = '{' + ids_data + '}'

        dist_fn = f'{self._wrapper.distance_function}_distance'

        init_knn_layer += f'''
        static {data_type} neighbors_features[] = {features_data};
        static uint16_t neighbors_id[] = {ids_data};
    '''

        init_knn_layer += f'''
        k_neighbors_classifier_layer_t layer= {{ n_neighbors, n_samples, n_features, n_classes, neighbors_features, neighbors_id {quant_params}, {dist_fn} }};
        return layer;
    }}
    '''
        return init_knn_layer

    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element.

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "knn.c".

        """
        return f'''k_neighbors_classifier_layer({self.name}_data, {input_name}, &{output_name});
    '''
