from tabnanny import verbose

from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)

import joblib as jl
import numpy as np

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'Prj-DenseNet'

# Crear datos sintéticos simples (4 entradas, valores entre 0 y 1)
np.random.seed(42)
X = np.random.rand(100, 4)  # 100 muestras, 4 características entre 0 y 1
y = (X.sum(axis=1) > 2).astype(int)  # Salida binaria basada en suma de entradas

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo MUY SIMPLE: 1 capa densa
model = Sequential([
    Input(shape=(4,)),
    Dense(1, activation='linear', name='capa_salida',
          kernel_initializer='ones', bias_initializer='zeros')  # 1 salida
])

# Configurar pesos manualmente a 1 (ya lo hace el inicializador 'ones')
# layer = model.layers[0]
# layer.set_weights([np.ones((4, 1)), np.zeros(1)])  # pesos y bias

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='mse',  # Error cuadrático medio para regresión
    metrics=['mse']
)

model._name = "simple_dense_model"
model.summary()

# Entrenamiento muy básico (realmente no necesario para este ejemplo)
history = model.fit(
    X_train, y_train,
    epochs=2,  # Pocas épocas ya que el modelo es trivial
    batch_size=10,
    validation_data=(X_test, y_test),
    verbose=1
)

options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'

# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.data_type = ModelDataType.FLOAT
#options.data_type = ModelDataType.FULL_QUANT8
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

# Usar datos de prueba como ejemplos
samples = X_test
ids = y_test
options.example_data = samples
options.example_ids = ids

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove output folder and start clean export
options.clean_output = True

############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)

# Inspección del modelo
from embedia.utils.model_inspector import ModelInspector

inspector = ModelInspector(model)

s_id = 0  # Primer ejemplo
for s_id, sample in enumerate(samples):
    weights, biases = model.layers[0].get_weights()
    print("Input  :", sample)
    print("Weights:",weights.flatten(), biases[0])
    print("Output :", model.predict(sample.reshape(1, -1), verbose=0)[0][0], "expected: ", ids[s_id])
    #print("Manual calculation (sum of inputs):", sample.sum())  # Debería ser igual