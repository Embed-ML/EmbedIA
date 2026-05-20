<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA is a machine learning framework for developing applications on microcontrollers.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

**Languages:** [English](README.md) | [Español](README_ES.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Italiano](README_IT.md) | [Português](README_PT.md) | [Русский](README_RU.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

EmbedIA is a compact and lightweight machine learning framework for deploying models on microcontrollers with limited hardware resources. It supports both neural network models (trained with TensorFlow/Keras) and machine learning algorithms (from Scikit-learn), enabling efficient inference execution on embedded systems. It is designed to be compatible with C and C++ languages for the Arduino IDE and supports a wide range of microcontrollers (MCUs).

## 📑 Table of Contents <A NAME="tabla-de-contenidos"></A>
* [Workflow](#workflow)
* [Layers](#layers)
* [Getting started](#started)
* [EmbedIA in C](#inC)


## 🔨 Workflow <A NAME="workflow"></A>
EmbedIA supports two types of machine learning models:

### 🧠 For Neural Networks (TensorFlow/Keras)
1. <strong>Model Generation:</strong> Select architecture, configure hyperparameters, and prepare training data.
2. <strong>Training:</strong> Train your Neural Network using TensorFlow/Keras in Python.
3. <strong>EmbedIA Export:</strong> Convert and export the model to C/C++ using the EmbedIA converter.
4. <strong>Deployment:</strong> Compile the project on your target microcontroller platform.
5. <strong>Inference:</strong> Run predictions on the embedded device.

### 🤖 For Machine Learning Models (Scikit-learn)
1. <strong>Model Training:</strong> Train classifiers or regressors using Scikit-learn.
2. <strong>EmbedIA Export:</strong> Convert the trained model to C/C++ using the EmbedIA converter.
3. <strong>Deployment:</strong> Compile the project on your target microcontroller platform.
4. <strong>Inference:</strong> Run predictions on the embedded device.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 🧅 Layers & Models <A NAME="layers"></A>
EmbedIA supports a comprehensive set of layers for neural networks and models from popular machine learning frameworks:

### ⚡ Neural Network Layers (TensorFlow/Keras)

**Convolutional Layers:**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Core Layers:**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Pooling Layers:**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Reshaping Layers:**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Normalization Layers:**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Activation Functions:**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Quantized Layers (Larq):**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 🎯 Machine Learning Models (Scikit-learn)

**Preprocessing:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Classification & Regression Models:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K-Nearest Neighbors Classifier</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K-Nearest Neighbors Regressor</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Support Vector Machine (SVM) Classifier</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Linear SVM Classifier</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Decision Tree Classifier</a>

### 🔊 Native Signal Processing Layers

* **STFT:** Short-Time Fourier Transform for 1D multi-spectrum analysis
* **Spectrogram:** Signal processing layer for audio and signal analysis

## 🚀 Getting started <A NAME="started"></A>
In order to use the EmbedIA Python converter, the first step is to clone the repository:

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Open the <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> script and configure the converter parameters. This script supports both TensorFlow/Keras models and Scikit-learn models:

**Core Parameters:**
* _OUTPUT_FOLDER_: output folder path
* _PROJECT_NAME_: generated project name
* _MODEL_FILE_: model path (.h5 format for Keras, or pickled model for Scikit-learn)

**Configuration Options:**
* _options.embedia_folder_: folder of EmbedIA files:
  * ```options.embedia_folder = ...```
* _options.project_type_: type of project among those available:
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: selection of microcontroller type among those available:
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: selection of data type among those available:
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: number of fractional bits for fixed-point data types (None for default):
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: options for block size of binary layers:
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: options for inclusion and use of debug functions:
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: Selection of files to be executed:
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: supported model to convert (TensorFlow/Keras, Scikit-Learn, etc.)
* _options.preprocessing_: list/object for preprocessing data (e.g.: normalization)
  * ```options.preprocessing_ = []```
* _options.example_data_: array of data to include as examples:
  * ```options.example_data = samples```
* _options.example_labels_: array of labels for examples (classification):
  * ```options.example_labels = labels```
* _options.baud_rate_: Arduino only, set Serial device speed:
  * ```options.baud_rate = 9600```
* _options.verbose_: verbose output during project generation:
  * ```options.verbose = True```
* _options.clean_output_: if True, remove output folder and start a clean export:
  * ```options.clean_output = True```
* _options.output_subfolder_: name of folder to store all embedia files:
  * ```options.output_subfolder = 'embedia'```

Run the script as follows:
```bash
python create_embedia_project.py
```

If the process was successful, a message will be displayed indicating where the project has been generated.

**Examples:**
* <strong>TensorFlow/Keras:</strong> Check the <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">Google Colab notebook</a> for a complete example of converting a CNN model trained on the MNIST dataset to C language.
* <strong>Simulation:</strong> Try the generated code online in the <a href="https://wokwi.com/projects/359745013247499265">Wokwi simulator</a>.


## 👍 EmbedIA in C/C++ <A NAME="inC"></A>
To use the EmbedIA features in the microcontroller, you need to include model initialization and inference execution in your code, using the provided functions:

* ```void model_init(void)```: Initializes the model in C language, loading the weights and parameters converted from your trained model (TensorFlow/Keras or Scikit-learn).
* ```int model_predict(input, * results)```: Executes inference using the input data passed as parameter. This function builds the complete model architecture by concatenating layer outputs in the correct order. It returns the prediction result and populates the results array with confidence scores or output values.

<strong>Example (Classification):</strong>
```c
// model initialization
model_init();

// model inference
int prediction = model_predict(input, &results);

// 'prediction' contains the predicted class ID
// 'results' contains the confidence scores for each class
```

<strong>Example (Regression or Multi-output):</strong>
```c
// model initialization
model_init();

// model inference - for regression or multi-output models
int status = model_predict(input, &results);

// 'results' contains the predicted values
```
