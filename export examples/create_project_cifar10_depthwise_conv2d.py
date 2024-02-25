import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

import joblib
from tensorflow.keras.models import load_model
from embedia.utils.model_inspector import ModelInspector
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
PROJECT_NAME = 'Prj-cifar10-DepthWiseConv2D_Net'
# MODEL_FILE = 'models/MNIST_14x14_model.h5'
MODEL_FILE = 'models/cifar10_depthwise_model.h5'

SAMPLES_FILE = 'samples/cifar10_20samples_16x16.sav'

#model = load_model(MODEL_FILE)
model = ModelLoader.load_model(MODEL_FILE)

model._name = "cifar10_model_dpw"

model.summary()

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

(samples, ids) = joblib.load(SAMPLES_FILE)


inspector = ModelInspector(model)
s_id = 0
# print(inspector.as_string(samples[s_id]))
sample = samples[s_id]

inspector.save(f'{model.name}_sample{s_id}.txt', sample, ln_break=-1)


res = model.predict(samples)

print((res*100).astype('int'))

options.example_data = samples
options.example_ids = ids

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove playground folder and start a clean export
options.clean_output = True

############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)

import larq

larq.models.summary(model)
