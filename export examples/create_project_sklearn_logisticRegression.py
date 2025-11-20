import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator


##############  Cración del modelo ###############

# Cargar el dataset Iris
data = load_iris()
X, y = data.data, data.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# copia para usar ejemplos originales en EmbedIA
X_test_raw = X_test.copy()
y_test_raw = y_test.copy()

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo KNN
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)

# Evaluar el modelo
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')


############# Settings to create the project #############

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'Prj-SKL_LogisticRegression_Iris'


model = lr 

samples = X_test_raw[0:10]
ids = y_test_raw[0:10]

options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
# options.debug_mode = DebugMode.DATA

options.files = ProjectFiles.ALL()
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

options.preprocessing = scaler

options.example_data = samples
options.example_ids = ids

############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)