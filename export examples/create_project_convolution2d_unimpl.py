import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

import os
import joblib
from tensorflow.keras.models import load_model
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType,
    UnimplementedLayerAction
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    SeparableConv2D,
    Flatten,
    BatchNormalization,
    Activation,
    ReLU,
    LeakyReLU,
    Softmax,
    MaxPooling2D,
    AveragePooling2D
)

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj_'
SAMPLES_FILE = 'samples/MNIST_20samples_12x12.sav'

model = Sequential()

model._name = "LayersTest01"

model.add(RandomFlip(mode='horizontal', input_shape=(12,12,1)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='LeakyReLU', name='Conv2D_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'))
model.add(LeakyReLU(name='LeakyReLU_1'))
model.add(BatchNormalization(name='Batch_1'))

model.add(Conv2D(16, kernel_size=(3, 3), activation='ReLU', name='Conv2D_2'))
model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool_2'))
model.add(ReLU(name='ReLU_2'))

model.add(SeparableConv2D(16, kernel_size=(3, 3), activation='sigmoid', name='SepConv2D_3'))
model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool_3'))

model.add(Flatten(name='Flatten_4'))
model.add(Dense(32, name='Dense_4', activation='tanh'))
model.add(Dense(8, name='Dense_5', activation='linear'))
model.add(Softmax(name='SoftMax_6'))


options = ProjectOptions()

# set location of EmbedIA folder
options.embedia_folder = '../embedia/'


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

# if True, remove playground folder and start a clean export
options.clean_output = True

# options.on_unimplemented_layer = UnimplementedLayerAction.FAILURE
# options.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_ALL
options.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_KNOWN



############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)

import larq

larq.models.summary(model)
