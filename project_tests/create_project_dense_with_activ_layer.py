import joblib as jl
import numpy as np
from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj-DenseNetWithActivationLayer'
MODEL_FILE = 'models/rain_predictor_model_with_activ_layer.h5'


# scaler = jl.load('scalers/rain_predictor_min_max_scaler.sav')
scaler = jl.load('scalers/rain_predictor_std_scaler.sav')


model = load_model(MODEL_FILE)

model._name = "rain_predictor"

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

# Normalizer not included in the model
options.normalizer = scaler

data = np.array([
    [34,-1,1005.63,24,4,43,26,9,-10,1009,999, 0],
    [36,4,1005.46,21,6,43,29,10,-2,1008,1001,0],
    [35,6,1006.00,27,5,41,29,12,-2,1009,1000,0],
    [34,7,1005.65,29,6,41,27,13,0,1008,1001,0],
    [31,11,1007.94,61,13,38,24,16,6,1011,1003,1],
    [28,13,1008.39,69,18,34,21,17,9,1011,1004,0],
    [30,10,1007.62,50,8,38,23,14,6,1010,1002,0],
    [34,8,1006.73,32,7,41,26,12,6,1010,1002,0],
    [34,11,1005.75,45,7,42,27,16,7,1008,1000,1],
    [34,16,1007.10,51,12,41,27,18,13,1010,1002,1],
    [32,16,1006.78,66,16,40,25,22,10,1011,1001,1],
    [34,13,1003.83,58,9,42,27,20,10,1007,998,0.1]
    ])

samples = data[:, 0:-1]
ids = data[:, -1]

options.example_data = samples
options.example_ids = ids

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove output folder and start clean export
options.clean_output = True

# ------------------ Generate project --------------------

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)
