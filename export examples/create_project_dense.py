from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from tensorflow.keras.models import Sequential
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
MODEL_FILE    = 'models/rain_predictor_model_with_activ_layer.h5'


#scaler = jl.load('scalers/rain_predictor_min_max_scaler.sav')
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

# options.data_type = ModelDataType.FLOAT
options.data_type = ModelDataType.QUANT8
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

# Normalizer not included in the model
options.normalizer = scaler

# meantempm,meandewptm,meanpressurem,maxhumidity,minhumidity,maxtempm,mintempm,maxdewptm,mindewptm,maxpressurem,minpressurem,date_sin,date_cos,rained
# date_sin = cos(2 * pi * day_of_year(date) / 365.25)
# date_cos = cos(2 * pi * day_of_year(date) / 365.25)
data = np.array([
    [34, -1, 1005.63, 24, 4, 43, 26, 9, -10, 1009, 999, 0.8638670987837349, -0.503719798736334, 0],
    [36, 4, 1005.46, 21, 6, 43, 29, 10, -2, 1008, 1001, 0.8550745085896932, -0.518505144391157, 0],
    [35, 6, 1006.0, 27, 5, 41, 29, 12, -2, 1009, 1000, 0.8460288880917339, -0.533137056031809, 0],
    [34, 7, 1005.65, 29, 6, 41, 27, 13, 0, 1008, 1001, 0.8367329140344921, -0.5476112038402312, 0],
    [31, 11, 1007.94, 61, 13, 38, 24, 16, 6, 1011, 1003, 0.8271893372462497, -0.5619233046832195, 1],
    [28, 13, 1008.39, 69, 18, 34, 21, 17, 9, 1011, 1004, 0.8174009818249189, -0.5760691233798758, 0],
    [30, 10, 1007.62, 50, 8, 38, 23, 14, 6, 1010, 1002, 0.8073707443023492, -0.5900444739548627, 0],
    [34, 8, 1006.73, 32, 7, 41, 26, 12, 6, 1010, 1002, 0.7971015927871943, -0.6038452208771035, 0],
    [34, 11, 1005.75, 45, 7, 42, 27, 16, 7, 1008, 1000, 0.7865965660866021, -0.6174672802835514, 1],
    [32, 16, 1006.78, 66, 16, 40, 25, 22, 10, 1011, 1001, 0.7648913904341225, -0.6441592666722685, 1],
    [18, 9, 1019.98, 77, 24, 26, 11, 12, 7, 1023, 1017, 0.38541335945432564, 0.9227440286201427, 0],
    [21, 12, 1018.49, 82, 29, 28, 14, 15, 11, 1022, 1015, 0.4012289854475009, 0.9159777842484877, 0],
    [22, 14, 1017.9, 77, 32, 27, 17, 17, 12, 1021, 1014, 0.4169258813287287, 0.908940487313808, 1],
    [16, 15, 1017.47, 100, 66, 18, 14, 16, 14, 1020, 1015, 0.4324994021339788, 0.901634220265486, 1]
    ])

samples = data[:, 0:-1]
ids = data[:, -1]
options.example_data = samples
options.example_ids = ids

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove playground folder and start clean export
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

print(sample)
print(scaler.transform([sample]))
new_sample = scaler.transform([sample])[0]
inspector.save('PruebaInspeccion'+f'sample{s_id}.txt', new_sample, ln_break=-1)


# print(model.get_config())

