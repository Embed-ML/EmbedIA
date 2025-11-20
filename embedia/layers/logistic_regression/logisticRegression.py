from embedia.core.logistic_regression_base_layer import LogisticRegressionBaseLayer
import numpy as np


class LogisticRegressionLayer(LogisticRegressionBaseLayer):
    """
    Esta es la clase "Generador" de la capa.
    Hereda de la BaseLayer (para obtener 'required_files')
    y usa el Wrapper (que se le "inyecta" como self._wrapper)
    para generar el código C.
    """

    def __init__(self, model, wrapper, **kwargs):
        # Llama al __init__ del padre (BaseLayer -> Layer)
        # Esto guarda self.model, self._wrapper, y self.name
        super().__init__(model, wrapper, **kwargs)
    
    @property
    def function_implementation(self):
        """
        Genera el CÓDIGO C de la función init() con los pesos
        del modelo entrenado.
        Automatización de lo hecho en 'model.c'
        """

        weights_flat = self._wrapper.weights.flatten() 
        bias = self._wrapper.bias
        classes = self._wrapper.classes
        
        weights_str = ",".join(f"{w:.6f}" for w in weights_flat)
        bias_str = ",".join(f"{b:.6f}" for b in bias)
        classes_str = ",".join(f"{c:.6f}" for c in classes)

        init_lr_layer = f""" 
static float {self.name}_weights[]={{ {weights_str} }};;
static float {self.name}_bias[]={{ {bias_str} }};
static float {self.name}_classes[] ={{ {classes_str} }};

logistic_regression_layer_t init_{self.name}_data(void)
{{
    logistic_regression_layer_t lr = {{
        .n_features= {self._wrapper.n_fetures},
        .n_classes= {self._wrapper.n_classes},
        .weights= {self.name}_weights,
        .bias= {self.name}_bias,
        .classes = {self.name}_classes,
    }};
    return lr;
}}
"""
        return init_lr_layer



    def invoke(self, input_name, output_name):
        """
        Genera la línea de C que llama a la función de predicción
        de la librería.
        """

        init_func = f"init_{self.name}_data()"


        return f"""

static logistic_regression_layer_t lr;
static int initialized = 0; // Bandera

if ({self.name}_initialized == 0) {{
        {self.name}_layer = {init_func};
        {self.name}_initialized= 1;
}}  

    logistic_regression_layer({self.name}_layer,{input_name}, {output_name};
"""

