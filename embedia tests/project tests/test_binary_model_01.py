
import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '../..')

import os
from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    BinaryBlockSize,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)
import joblib
import compilers
import larq as lq

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj-CIFAR10_QuantSepConv2D_'
MODEL_FILE = 'models/CIFAR10_QuantSeparableConv2D_50acc.h5'
SAMPLES_FILE = 'samples/CIFAR10_20samples_32x32.sav'


model = load_model(MODEL_FILE)

model._name = "QuantSeparableConv2DTest"

model.summary()

try:
    import larq

    larq.models.summary(model)
except Exception:
    pass


options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../../embedia/'

options.data_type = ModelDataType.BINARY
#options.data_type = ModelDataType.BINARY_FIXED32
#options.data_type = ModelDataType.BINARY_FLOAT16

#options.tamano_bloque = BinaryBlockSize.Bits8
#options.tamano_bloque = BinaryBlockSize.Bits16
options.tamano_bloque = BinaryBlockSize.Bits32
#options.tamano_bloque = BinaryBlockSize.Bits64

options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
#options.debug_mode = DebugMode.DATA


(samples, ids) = joblib.load(SAMPLES_FILE)

options.example_data = samples
options.example_ids = ids

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
    (ProjectType.CODEBLOCK, ModelDataType.BINARY, 'CB_BIN'),
    (ProjectType.CODEBLOCK, ModelDataType.BINARY_FIXED32, 'CB_BF32'),
    #(ProjectType.CODEBLOCK, ModelDataType.BINARY_FLOAT16, 'CB_BF16'),
    (ProjectType.ARDUINO, ModelDataType.BINARY, 'AR_BIN'),
    (ProjectType.ARDUINO, ModelDataType.BINARY_FIXED32, 'AR_BF32'),
    #(ProjectType.ARDUINO, ModelDataType.BINARY_FLOAT16, 'AR_BF16')
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
