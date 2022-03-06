from tensorflow.keras.models import load_model
from sklearn.datasets import load_digits

from embedia.project_options import *
from embedia.project_generator import ProjectGenerator


############# Configuracion para crear el proyecto #############

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'C_mnist_min_tanh_float'
MODEL_FILE    = 'models/mnist_model_min_tanh.h5'

model = load_model(MODEL_FILE)

model._name = 'MNIST_Digits'

digits = load_digits()
example_number = 33
sample = digits.images[example_number]
comment= "number %d example for test" % digits.target[example_number]

options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
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

options.example_data = sample
options.example_comment = comment

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

generator = ProjectGenerator()
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("\nProyecto", PROJECT_NAME, "exportado en", OUTPUT_FOLDER)

