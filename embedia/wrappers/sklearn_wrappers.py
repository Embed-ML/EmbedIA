from embedia.core.layer_wrapper import LayerWrapper

class ScikitLearnWrapper(LayerWrapper):
    @property
    def name(self):
        return self._target.__class__.name


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
