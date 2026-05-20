import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelMicro,
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)
from embedia.utils.model_loader import ModelLoader

##############################################################################
# Paquetes sklearn para generación de un modelo para SVM con el dataset Iris #

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris
data = load_iris()
X, y = data.data, data.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# copia para usar ejemplos originales en EmbedIA
X_test_raw = X_test.copy()

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo SVM
svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
svm.fit(X_train, y_train)

# Evaluar el modelo
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'Prj-SKL_SVM_Iris'

model = svm
model.name = "SKL_SVM_iris_model"

options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'

# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.micro = ModelMicro.GENERIC
#options.micro = ModelMicro.ESP32



#options.data_type = ModelDataType.FLOAT
#options.data_type = ModelDataType.FULL_QUANT8 # revisar sigmoid activation y demas
#options.data_type = ModelDataType.FIXED32
options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8
# options.data_type = ModelDataType.QUANT8
# for fixed point data types, set the number of bits for the fractional part (e.g., 16 for Q16.16)
#options.fixed_precision = 7  # for FIXED data types


# options.debug_mode = DebugMode.DISCARD
options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
# options.debug_mode = DebugMode.DATA

(samples, ids) = (X_test_raw, y_test)

i=0
for i in range(10):
    print(X_test_raw[i],X_test[i], y_test[i], model.predict([X_test[i]])[0])

#print(model.decision_function(X_test))

options.example_data = samples
options.example_ids = ids
options.preprocessing = scaler
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