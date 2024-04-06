
import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '../..')

import os
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)
from supported_layers_model import build_all_layers_model

import compilers

# keras.layers.SeparableConv2D: SeparableConv2D,
# keras.layers.Conv2D: Conv2D,
# keras.layers.Dense: Dense,
# keras.layers.Flatten: Flatten,
# keras.layers.BatchNormalization: BatchNormalization,
# keras.layers.Activation: Activation,
# keras.layers.ReLU: Activation,
# keras.layers.LeakyReLU: Activation,
# keras.layers.Softmax: Activation,
# # pooling layers
# keras.layers.AveragePooling1D: Pooling,  # not yet implemented in C
# keras.layers.AveragePooling2D: Pooling,
# keras.layers.AveragePooling3D: Pooling,  # not yet implemented in C
# keras.layers.MaxPooling1D: Pooling,      # not yet implemented in C
# keras.layers.MaxPooling2D: Pooling,
# keras.layers.MaxPooling3D: Pooling,      # not yet implemented in C

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj_Test_'
SAMPLES_FILE = 'samples/MNIST_20samples_12x12.sav'

(model, x_test, y_test) = build_all_layers_model()

model._name = "LayersTest01"

model.summary()

try:
    import larq

    larq.models.summary(model)
except Exception:
    pass


options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../../embedia/'


options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

# options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
options.debug_mode = DebugMode.DATA

# (samples, ids) = joblib.load(SAMPLES_FILE)

# options.example_data = samples
# options.example_ids = ids

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

# if True, remove output folder and start a clean export
options.clean_output = True

# ############ Generate project and test project #############

# get the current directory path
current_dir = os.getcwd()

# configuration of each project to export
PROJECT_LIST = [
    (ProjectType.ARDUINO, ModelDataType.FLOAT, 'AR_FLT'),
    (ProjectType.ARDUINO, ModelDataType.FIXED32, 'AR_F32'),
    (ProjectType.ARDUINO, ModelDataType.FIXED16, 'AR_F16'),
    (ProjectType.ARDUINO, ModelDataType.FIXED8, 'AR_F8'),
    (ProjectType.ARDUINO, ModelDataType.QUANT8, 'AR_Q8'),
    (ProjectType.CODEBLOCK, ModelDataType.FLOAT, 'CB_FLT'),
    (ProjectType.CODEBLOCK, ModelDataType.FIXED32, 'CB_F32'),
    (ProjectType.CODEBLOCK, ModelDataType.FIXED16, 'CB_F16'),
    (ProjectType.CODEBLOCK, ModelDataType.FIXED8, 'CB_F8'),
    (ProjectType.CODEBLOCK, ModelDataType.QUANT8, 'CB_Q8')
    ]

results = []
for (p_type, d_type, p_name) in PROJECT_LIST:
    options.project_type = p_type
    options.data_type = d_type

    header_text = f' Project {PROJECT_NAME + p_name} exported in {OUTPUT_FOLDER} '
    print('\n'+'#'*100)
    print(header_text.center(100, '#'))
    print('#'*100)

    prj_folder = PROJECT_NAME+p_name
    generator = ProjectGenerator(options)
    generator.create_project(OUTPUT_FOLDER, prj_folder, model, options)

    if p_type == ProjectType.CODEBLOCK:
        # construct the full path to the project file
        project_path = os.path.join(current_dir, OUTPUT_FOLDER, prj_folder,prj_folder+'.cbp')

        print('Compiling %s for Code::Blocks: ' % prj_folder, end='')
        result = compilers.codeblocks_compile(project_path)

    elif p_type == ProjectType.ARDUINO:
        # construct the full path to the project file
        project_path = os.path.join(current_dir, OUTPUT_FOLDER, prj_folder, prj_folder+'.ino')

        print('Compiling %s for Arduino: ' % prj_folder, end='')
        result = compilers.arduino_cli_compile(project_path, board='esp32:esp32:nodemcu-32s')

    results.append((prj_folder, result))

    print(result[1].name+'\n')  #+'\n'+'#'*100+'\n')

print('\nTest Summary:')
for (prj_name, result) in results:
    print(prj_name, result[1].name)
