import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '../..')
from embedia.utils.converters  import tflite
import tensorflow as tf

from supported_layers_model import build_all_layers_model

TFLITE_FILE = 'full_model.tflite'
TF_OUTFILE = 'full_model_reconverted.h5'

model = build_all_layers_model()

model._name = "Layers_tflite"

model.summary()

tfl_converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Configurar las opciones de conversión
# tfl_converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tfl_converter.target_spec.supported_types = [tf.float16]

# Convertir el modelo y guardar en disco
tflite_model = tfl_converter.convert()
open('models/'+TFLITE_FILE, "wb").write(tflite_model)

# levantar modelo y reconvertirlo
converter = tflite.load_model('models/'+TFLITE_FILE)
tfl_model = tflite.convert_to_tf(converter)


tfl_model.build()
tfl_model.summary()

tfl_model.save('models/'+TF_OUTFILE)
