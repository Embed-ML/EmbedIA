import numpy as np
from tensorflow.keras.models import load_model
from sklearn.datasets import load_digits

from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator

############# Settings to create the project #############

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'C_mnist_float'
MODEL_FILE    = 'models/mnist_model.h5'

model = load_model(MODEL_FILE)

digits = load_digits()
samples = digits.images[0:10]
ids = digits.target[0:10]

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

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

options.example_data = samples
options.example_ids = ids

############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)