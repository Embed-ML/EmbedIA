import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

import joblib
import numpy as np
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



from tensorflow.keras import models, layers
OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj-ZeroPadding-Test'

model = models.Sequential()
model.add(layers.ZeroPadding2D(padding=(2,1), input_shape=(3,3,3)))


model._name = "zero_padding2d_test"

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
# options.data_type = ModelDataType.QUANT8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove output folder and start a clean export
options.clean_output = True

samples = np.arange(1, 28).reshape((1, 3, 3, 3))

options.example_data = samples
options.example_ids = np.array([0])
############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)

import larq

larq.models.summary(model)

inspector = ModelInspector(model)
s_id = 0
# print(inspector.as_string(samples[s_id]))
sample = samples[s_id]

inspector.save(f'{OUTPUT_FOLDER}/{PROJECT_NAME}/{model.name}_sample{s_id}.txt', sample, ln_break=-1)

