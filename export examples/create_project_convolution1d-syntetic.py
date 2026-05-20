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
PROJECT_NAME = 'Prj-Conv1D_Net_synt'


tf.random.set_seed(0)

# Modelo simple: una capa Conv1D
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8, 1)),  # 8 timesteps, 1 canal
    tf.keras.layers.Conv1D(
        filters=2,       # 2 filtros
        kernel_size=2,   # tamaño del kernel
        strides=1,
        padding="same", # sin padding
        use_bias=True
    ),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation=None)
])

weights = model.layers[0].get_weights()
print("Kernel shape:", weights[0].shape)  # (kernel_size, in_channels, filters)
print("Bias shape:", weights[1].shape)

# Creamos una entrada simple
x = np.arange(8, dtype=np.float32).reshape(1, 8, 1)  # batch=1, secuencia=8
print("Input:", x[0, :, 0])

# Ejecutamos el modelo
y = model.predict(x)
print("Output:", y[0])


model._name = "model_conv1D"

#model.summary()

options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'


# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP
#options.project_type = ProjectType.CMAKE_C
#options.project_type = ProjectType.CMAKE_CPP

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




