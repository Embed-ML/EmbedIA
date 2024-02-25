import numpy as np
import pickle
import pathlib, os
import librosa
import tensorflow as tf
import urllib.request, zipfile
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from embedia.utils.melspec import Melspec
from embedia.model_generator.project_options import *
from embedia.project_generator import ProjectGenerator
from embedia.layers.signal_processing.spectrogram import Spectrogram

# ---------------------------------------------- FUNCIONES ---------------------------------------------- #
fs = 6000
PATH = 'mini_speech_commands'
# Cargado del dataset
def create_dataset(fs, PATH):
    y = []
    x = []

    data_dir = pathlib.Path(PATH)
    if not data_dir.exists():
        # tf.keras.utils.get_file(
        #     'mini_speech_commands.zip',
        #     origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        #     extract=True,
        #     cache_dir='.', cache_subdir='data')

        url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
        urllib.request.urlretrieve(url, "mini_speech_commands.zip")
        with zipfile.ZipFile("mini_speech_commands.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    # labels = ['yes', 'no', 'up', 'down', 'right', 'left', 'stop', 'go']
    # labels = ['yes', 'no', 'up', 'down', 'right', 'left','go']
    labels = ['yes', 'no', 'up', 'down', 'right', 'left']

    print('Labels:', labels)

    for (y_value, com) in enumerate(labels):
        for file_name in os.listdir(os.path.join(data_dir,com)):
            file_path = os.path.join(data_dir,com+'/'+file_name)
            data, _ = librosa.load(file_path, sr=fs)
            data = np.array(data)
            data = np.append(data,[0]*(fs-len(data)))

            y.append(y_value)
            x.append(data)

    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    return (x_train,y_train), (x_test,y_test), labels

def preprocess_data(melspec, x_train, x_test):
    # spec_x_train = np.zeros((len(x_train),windows_t,windows_f,1), dtype=np.float32)
    # spec_x_test = np.zeros((len(x_test),windows_t,windows_f,1), dtype=np.float32)

    # Mejor utilizar esta opción por ahora
    spec_x_train = np.zeros((len(x_train),melspec.n_blocks,melspec.n_mels,1), dtype=np.float32)
    spec_x_test = np.zeros((len(x_test),melspec.n_blocks,melspec.n_mels,1), dtype=np.float32)

    print('Preparando train...')
    for i_train in range(len(x_train)):
        dato = np.float32(x_train[i_train])
        mfcc = melspec.process_melspectrogram(dato,fs)
        mfcc = np.reshape(mfcc, mfcc.shape + (1,))
        spec_x_train[i_train] = mfcc

    print('Preparando test...\n')
    for i_test in range(len(x_test)):
        dato = np.float32(x_test[i_test])
        mfcc = melspec.process_melspectrogram(dato,fs)
        mfcc = np.reshape(mfcc, mfcc.shape + (1,))
        spec_x_test[i_test] = mfcc

    return spec_x_train , spec_x_test, melspec



# ---------------------------------------------- PROGRAMA PRINCIPAL ---------------------------------------------- #
############# Settings to create the project #############

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'C_mnist_fixed_SPEC'
MODEL_FILE    = 'models/spec_model.h5'

model = load_model(MODEL_FILE)


(x_train,y_train), (x_test,y_test), labels = create_dataset(fs, PATH)

# Preprocesamiento de los datos
n_mels = 32 # Modicar a 23 para que la imagen sea cuadrada prueba imagen rectangular para ver si solucionó el problema
# n_mels = 64
n_fft = 256
# n_fft = 128

noverlap = 0
largo = x_train[0].shape[0]
windows_t = 23 #no tocar. Imagen cuadrada
windows_f = 32

samples = np.array(x_test[:10])
ids = y_test[:10]

melspec = Melspec(n_fft, n_mels, noverlap)
melspec.calculate_params(largo, fs, report=True)

spec_x_train, spec_x_test, melspec = preprocess_data(melspec, x_train,x_test)
print('spec_x_train.shape:', spec_x_train.shape)
print('spec_x_test.shape: ', spec_x_test.shape)

# with open("melspec.pickle", "wb") as f:
#     pickle.dump(melspec, f)

samples = np.array(x_test[:10])
ids = y_test[:10]

options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

# options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
options.data_type = ModelDataType.FIXED16
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

# with open("melspec.pickle", "rb") as f:
#     options.normalizer = pickle.load(f)

options.normalizer = melspec
options.normalizer.report()

############# Generate project #############

generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)


# import matplotlib.pyplot as plt

# plt.plot(spec_x_test[0].flatten())
# plt.show()

# print("Shape:", spec_x_train[0].shape)