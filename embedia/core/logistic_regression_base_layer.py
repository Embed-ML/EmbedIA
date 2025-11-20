# logistic_regression_base_layer.py

from embedia.core.layer import Layer

class LogisticRegressionBaseLayer(Layer):
  """
  Clase base para la capa de Regresión Logística.
  Define los archivos .c y .h que necesita esta capa.
  """
  def __init__(self, model, wrapper, **kwargs):
    super().__init__(model, wrapper, **kwargs)

  @property
  def required_files(self):
    """
    Retorna una lista de tuplas con los archivos C y H
    que necesita esta capa para funcionar.
    """
    # (nombre_del_header.h, nombre_del_source.c)
    return super().required_files + [('logisticRegression.h', 'logisticRegression.c')]