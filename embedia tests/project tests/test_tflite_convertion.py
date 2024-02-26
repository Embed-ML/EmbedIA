import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '../..')
from tflite_convertion import *


tflite_input = 'model.tflite'


converter = load_tflite_model(tflite_input)
model = convert_tflite_to_tf(converter)


model.build()
model.summary()

