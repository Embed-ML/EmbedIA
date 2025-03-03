from embedia.core.layer_wrapper import LayerWrapper


class ScikitLearnWrapper(LayerWrapper):
    #@property
    #def name(self):
    #    return self._target.__class__.name
    pass


class SKLNormWrapper(ScikitLearnWrapper):
    @property
    def div_values(self):
        return None

    @property
    def sub_values(self):
        return None

    @property
    def input_shape(self):
        return self.div_values.shape

    @property
    def output_shape(self):
        return self.div_values.shape


class SKLMinMaxScalerWrapper(SKLNormWrapper):

    @property
    def div_values(self):
        return self._target.data_range_

    @property
    def sub_values(self):
        return self._target.data_min_

    @property
    def funcion_name(self):
        return 'robust'


class SKLMaxAbsScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.max_abs_

    @property
    def funcion_name(self):
        return 'max_abs'


class SKLStandardScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.scale_

    @property
    def sub_values(self):
        return self._target.mean_

    @property
    def funcion_name(self):
        return 'standard'


class SKLRobustScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.scale_

    @property
    def sub_values(self):
        return self._target.center_

    @property
    def funcion_name(self):
        return 'robust'


class SKLKnnWrapper(ScikitLearnWrapper):
    SUPPORTES_DISTANCES = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'braycurtis', 'canberra']

    @property
    def n_classes(self):
        return len(self._target.classes_)

    @property
    def n_neighbors(self):
        return self._target.n_neighbors

    @property
    def n_samples(self):
        return self._target.n_samples_fit_

    @property
    def n_features(self):
        return self._target.n_features_in_

    @property
    def fit_x(self):
        return self._target._fit_X

    @property
    def y(self):
        return self._target._y

    @property
    def input_shape(self):
        return self._target._fit_X[0].shape

    @property
    def output_shape(self):
        return (None, len(self._target.classes_))


    def _uniform_function_name(self, dist_name, extra_param):
        dist_name = dist_name.lower()
        if (dist_name == 'manhattan' or dist_name == 'cityblock' or
                dist_name == 'l1' or (dist_name == 'minkowski' and extra_param == 1)):
            return 'manhattan'
        if dist_name == 'euclidean' or dist_name == 'l2' or (dist_name == 'minkowski' and extra_param == 2):
            return 'euclidean'

        return dist_name

    @property
    def distance_function(self):

        extra_param = self._target.p
        fn_name = self._target.metric

        fn_name = self._uniform_function_name(fn_name, extra_param)
        if fn_name not in self.SUPPORTES_DISTANCES or (fn_name=='minkowski' and extra_param not in [1,2]):
            raise Exception('Unknown distance function for KNN algorithm')

        return fn_name

    @property
    def activation(self):
        return None

