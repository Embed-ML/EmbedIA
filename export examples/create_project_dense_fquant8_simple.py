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

iris = load_iris()
X, y = iris.data, iris.target

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear modelo SÚPER SIMPLE: solo 2 capas densas
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu', name='capa_oculta'),
    Dense(3, activation='softmax', name='capa_salida')  # 3 clases
])

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Para etiquetas enteras
    metrics=['accuracy']
)

model._name = "fquant8_prb"
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
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
# options.data_type = ModelDataType.FULL_QUANT8
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

# Normalizer not included in the model
# options.normalizer = scaler

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



from embedia.utils.model_inspector import ModelInspector

inspector = ModelInspector(model)

s_id = 4
# print(inspector.as_string(samples[s_id]))

sample = samples[s_id]
#sample =  np.array([sample[:, :,0], sample[:, :, 1]])

#print(sample)
#print(scaler.transform([sample]))
#new_sample = scaler.transform([sample])[0]
#inspector.save('PruebaInspeccion'+f'sample{s_id}.txt', new_sample, ln_break=-1)


# print(model.get_config())

s_id = 0  # Primer ejemplo
for s_id, sample in enumerate(samples):
    weights, biases = model.layers[0].get_weights()
    print("Input  :", sample)
    print("Weights:",weights.flatten(), biases[0])
    print("Output :", model.predict(sample.reshape(1, -1), verbose=0)[0][0], "expected: ", ids[s_id])
    #print("Manual calculation (sum of inputs):", sample.sum())  # Debería ser igual

