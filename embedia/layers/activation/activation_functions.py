from embedia.model_generator.project_options import ModelDataType
import re

class ActivationFunctions:

    def __init__(self, model, wrapper, **kwargs):
        self._model = model
        self._wrapper = wrapper

    def _base_params(self, output_name, output_size, qparams=''):
        if qparams != '':
            qparams = ', ' + qparams
        return f'{output_name}{qparams}, {output_size}'

    def _tanh_params(self, output_name, output_size, qparams=''):
        return self._base_params(output_name, output_size, qparams)

    def _sigmoid_params(self, output_name, output_size, qparams=''):
        return self._base_params(output_name, output_size, qparams)

    def _softmax_params(self, output_name, output_size, qparams=''):
        return self._base_params(output_name, output_size, qparams)

    def _softsign_params(self, output_name, output_size, qparams=''):
        return self._base_params(output_name, output_size, qparams)

    def _relu_params(self, output_name, output_size, qparams=''):
        return self._base_params(output_name, output_size, qparams)

    def _leakyrelu_params(self, output_name, output_size, qparams=''):
        if self._model.is_data_quantized: # no make sense to use quantization, use float
            (data_type, data_converter) = self._model.get_type_converter(ModelDataType.FLOAT)
        else: # default data type converter
            (data_type, data_converter) = self._model.get_type_converter()

        alpha = data_converter.fit_transform([self._wrapper.leakyrelu_alpha])
        extra_param = f', {alpha[0]}'
        return self._base_params(output_name, output_size, qparams) + extra_param

    def get_params(self, fnc_name, output_name, output_size, qparams=''):
        method_name = f'_{fnc_name}_params'
        method = getattr(self, method_name)
        return method(output_name, output_size, qparams)


    def invoke(self, output_name, var_output_size, qparams=''):
        fnc_name = self._wrapper.function_name
        if fnc_name == 'linear':
            return ''
        params = self.get_params(fnc_name, output_name, var_output_size, qparams)
        return f'{fnc_name}_activation({params});'
