from embedia.core.knn_base_layer import KnnBaseLayer
from embedia.model_generator.project_options import ModelDataType


class KNeighborsClassifier(KnnBaseLayer):
    support_quantization = True  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    @property
    def function_implementation(self):
        name = self.name
        struct_type = self.struct_data_type
        data_fit = self._wrapper.fit_data
        dist_fn = f'{self._wrapper.distance_function}_distance'
        is_mixed_type = (self.options.data_type == ModelDataType.QUANT8)

        (data_type, data_converter) = self.model.get_type_converter()
        conv_data_fit = data_converter.fit_transform(data_fit)

        if is_mixed_type:
            params = data_converter.export_params(mode="q15")
            qp_param = f',{{ {params.scale_q}, {params.zero_point} }}'
        else:
            qp_param = ''



        cb = self.c_builder

        with cb.bgn(f'{struct_type} init_{name}_data(void)'):
            cb.add(f'uint16_t n_neighbors = {self.wrapper.n_neighbors};')
            cb.add(f'uint32_t n_samples   = {self.wrapper.n_samples};')
            cb.add(f'uint16_t n_features  = {self.wrapper.n_features};')
            cb.add(f'uint16_t n_classes   = {self.wrapper.n_classes};')
            cb.add()

            cb.add_array(
                dtype=f'static {data_type}',
                name='neighbors_features',
                values=[v for row in conv_data_fit for v in row],
                cols=self.wrapper.n_features,
                comments=[str(cls) for cls in self.wrapper.fit_target],
            )
            cb.add()

            cb.add_array(
                dtype='static uint16_t',
                name='neighbors_id',
                values=self._wrapper.fit_target,
            )
            cb.add()

            fields = [
                'n_neighbors, n_samples, n_features, n_classes',
                f'neighbors_features, neighbors_id',
                f'{dist_fn}{qp_param}'
            ]
            cb.add_struct(
                dtype='k_neighbors_classifier_layer_t',
                name='layer',
                fields=fields,
            )
            cb.add()
            cb.add('return layer;')

        return cb.get_code()

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
