import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

import joblib
from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)
import numpy as np

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj-GlobalPool2D_Net_synt'


tf.random.set_seed(0)

# Modelo más simple para entender GlobalAveragePooling2D
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 3, 2)),  # 3x3 píxeles, 2 canales
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(1, activation=None)
])

# Creamos una entrada con valores específicos para entender el pooling
x = np.array([[
    [[1, 10], [2, 20], [3, 30]],
    [[4, 40], [5, 50], [6, 60]],
    [[7, 70], [8, 80], [9, 90]]
]], dtype=np.float32)  # batch=1, 3x3, 2 canales

print("Input shape:", x.shape)
print("Input - Canal 0:")
print(x[0, :, :, 0])
print("Input - Canal 1:")
print(x[0, :, :, 1])

# Calculamos manualmente lo que debería hacer GlobalAveragePooling2D
manual_pooling = np.mean(x[0], axis=(0, 1))
print("Manual GlobalAvg/MaxPooling2D result:", manual_pooling)

# Ejecutamos el modelo
y = model.predict(x)
print("Model output:", y[0])


model._name = "model_conv1D"

#model.summary()

options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'


# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

(samples, ids) = (x, y)

res = model.predict(samples, verbose=0)

print("Prediccion:", res)

# Salida de la conv1D
conv_layer = model.layers[0]
conv_out = conv_layer(x)  # shape = (1, 6, 2) -> batch, width, channels

# Quitamos batch dimension
conv_out = np.array(conv_out[0])  # shape = (6, 2)

# Rearreglamos para imprimir como [canal][tiempo]
conv_out_ch_major = conv_out.T  # shape = (2,6) -> canal primero

print("Conv1D output (canal-major):")
for c in range(conv_out_ch_major.shape[0]):
    print("Channel", c, ":", conv_out_ch_major[c])


# sample =  np.array([sample[:,:,0], sample[:,:,1]])


options.example_data = np.array(samples)
options.example_ids = np.array(ids)

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove output folder and start a clean export
options.clean_output = True


############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)


