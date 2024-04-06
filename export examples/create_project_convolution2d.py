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
from embedia.utils.model_loader import ModelLoader

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'ExampleProject'
PROJECT_NAME = 'Prj-Conv2D_Net'
# MODEL_FILE = 'models/MNIST_14x14_model.h5'
MODEL_FILE = 'models/MNIST[In14x14x1_C2(8)_MP2_C2(16)_MP2_Fl_De(16)_De(10)](1818)_94.49.tflite'

SAMPLES_FILE = 'samples/MNIST_20samples_14x14.sav'

#model = load_model(MODEL_FILE)
model = ModelLoader.load_model(MODEL_FILE)

model._name = "mnist_model"

model.summary()

options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'


# options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

#options.data_type = ModelDataType.FLOAT
options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

(samples, ids) = joblib.load(SAMPLES_FILE)

res = model.predict(samples)
print( (res*100).astype('int') )


options.example_data = samples
options.example_ids = ids

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

import larq

larq.models.summary(model)
