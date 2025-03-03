from embedia.core.knn_base_layer import KnnBaseLayer
import numpy as np


class KNeighborsRegressor(KnnBaseLayer):
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

        init_knn_layer = f'''
    {struct_type} init_{name}_data(void){{
    
        uint16_t n_neighbors = {self._wrapper.n_neighbors};
        uint32_t n_rows = {self._wrapper.n_samples_fit};
        uint16_t n_features = {self._wrapper.n_features_in};
    '''

        features_data = "\n".join(["{" + ", ".join(map(str, row)) + "}," for row in self._wrapper.fit_x[:-1]])
        features_data += "\n {" + ", ".join(map(str, self._wrapper.fit_x[-1])) + "}"
        features_data = '{' + features_data + '}'

        labels_data = ','.join([str(y) for y in self._wrapper.y])
        labels_data = '{' + labels_data + '}'

        neighbors_def = ''''''

        init_knn_layer += f'''
        static float neighbors_features[]{neighbors_def}[{self._wrapper.n_features_in}] = {features_data};
        static float neighbors_labels[] = {labels_data};
    '''

        init_knn_layer += f'''
        k_neighbors_regressor_layer_t layer= {{ n_neighbors, n_rows, n_features, *neighbors_features, neighbors_labels}};
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
        return f'''k_neighbors_regressor_layer({self.name}_data, {input_name}, &{output_name});
    '''
