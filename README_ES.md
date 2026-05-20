<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA es un framework de machine learning para desarrollar aplicaciones en microcontroladores.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

EmbedIA es un framework de machine learning compacto y ligero para desplegar modelos en microcontroladores con recursos de hardware limitados. Soporta tanto modelos de redes neuronales (entrenados con TensorFlow/Keras) como algoritmos de machine learning (de Scikit-learn), permitiendo la ejecución eficiente de inferencia en sistemas embebidos. Está diseñado para ser compatible con los lenguajes C y C++ para el Arduino IDE y soporta una amplia gama de microcontroladores (MCUs).

## Tabla de Contenidos <A NAME="tabla-de-contenidos"></A>
* [Flujo de trabajo](#workflow)
* [Capas](#layers)
* [Primeros pasos](#started)
* [EmbedIA en C](#inC)


## Flujo de trabajo <A NAME="workflow"></A>
EmbedIA soporta dos tipos de modelos de machine learning:

### Para Redes Neuronales (TensorFlow/Keras)
1. <strong>Generación del modelo:</strong> Seleccionar arquitectura, configurar hiperparámetros y preparar datos de entrenamiento.
2. <strong>Entrenamiento:</strong> Entrenar la Red Neuronal usando TensorFlow/Keras en Python.
3. <strong>Exportación con EmbedIA:</strong> Convertir y exportar el modelo a C/C++ usando el conversor de EmbedIA.
4. <strong>Despliegue:</strong> Compilar el proyecto en la plataforma de microcontrolador objetivo.
5. <strong>Inferencia:</strong> Ejecutar predicciones en el dispositivo embebido.

### Para Modelos de Machine Learning (Scikit-learn)
1. <strong>Entrenamiento del modelo:</strong> Entrenar clasificadores o regresores usando Scikit-learn.
2. <strong>Exportación con EmbedIA:</strong> Convertir el modelo entrenado a C/C++ usando el conversor de EmbedIA.
3. <strong>Despliegue:</strong> Compilar el proyecto en la plataforma de microcontrolador objetivo.
4. <strong>Inferencia:</strong> Ejecutar predicciones en el dispositivo embebido.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## Capas y Modelos <A NAME="layers"></A>
EmbedIA soporta un conjunto comprehensivo de capas para redes neuronales y modelos de frameworks populares de machine learning:

### Capas de Redes Neuronales (TensorFlow/Keras)

**Capas Convolucionales:**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Capas Principales:**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Capas de Pooling:**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Capas de Reorganización:**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Capas de Normalización:**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Funciones de Activación:**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Capas Cuantizadas (Larq):**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### Modelos de Machine Learning (Scikit-learn)

**Preprocesamiento:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Modelos de Clasificación y Regresión:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Clasificador K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">Regresor K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Clasificador Support Vector Machine (SVM)</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Clasificador SVM Lineal</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Clasificador Árbol de Decisión</a>

### Capas Nativas de Procesamiento de Señales

* **STFT:** Transformada de Fourier de Tiempo Corto para análisis multi-espectral 1D
* **Espectrograma:** Capa de procesamiento de señales para análisis de audio y señales

## Primeros Pasos <A NAME="started"></A>
Para usar el conversor Python de EmbedIA, el primer paso es clonar el repositorio:

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Abrir el script <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> y configurar los parámetros del conversor. Este script soporta tanto modelos TensorFlow/Keras como modelos Scikit-learn:

**Parámetros Principales:**
* _OUTPUT_FOLDER_: ruta de la carpeta de salida
* _PROJECT_NAME_: nombre del proyecto generado
* _MODEL_FILE_: ruta del modelo (formato .h5 para Keras, o modelo serializado para Scikit-learn)

**Opciones de Configuración:**
* _options.embedia_folder_: carpeta de archivos EmbedIA:
  * ```options.embedia_folder = ...```
* _options.project_type_: tipo de proyecto entre los disponibles:
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: selección de tipo de microcontrolador entre los disponibles:
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: selección de tipo de dato entre los disponibles:
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: número de bits fraccionales para tipos de datos de punto fijo (None para predeterminado):
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: opciones para tamaño de bloque de capas binarias:
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: opciones para inclusión y uso de funciones de depuración:
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: Selección de archivos a ejecutar:
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: modelo soportado para convertir (TensorFlow/Keras, Scikit-Learn, etc.)
* _options.preprocessing_: lista/objeto para preprocesamiento de datos (ej: normalización)
  * ```options.preprocessing_ = []```
* _options.example_data_: arreglo de datos para incluir como ejemplos:
  * ```options.example_data = samples```
* _options.example_labels_: arreglo de etiquetas para ejemplos (clasificación):
  * ```options.example_labels = labels```
* _options.baud_rate_: Solo para Arduino, configurar velocidad del dispositivo Serial:
  * ```options.baud_rate = 9600```
* _options.verbose_: salida detallada durante la generación del proyecto:
  * ```options.verbose = True```
* _options.clean_output_: si es True, elimina la carpeta de salida e inicia una exportación limpia:
  * ```options.clean_output = True```
* _options.output_subfolder_: nombre de la carpeta para almacenar todos los archivos embedia:
  * ```options.output_subfolder = 'embedia'```

Ejecutar el script de la siguiente manera:
```bash
python create_embedia_project.py
```

Si el proceso fue exitoso, se mostrará un mensaje indicando dónde se ha generado el proyecto.

**Ejemplos:**
* <strong>TensorFlow/Keras:</strong> Consultar el <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">notebook de Google Colab</a> para un ejemplo completo de conversión de un modelo CNN entrenado en el dataset MNIST a lenguaje C.
* <strong>Simulación:</strong> Probar el código generado en línea en el <a href="https://wokwi.com/projects/359745013247499265">simulador Wokwi</a>.


## EmbedIA en C/C++ <A NAME="inC"></A>
Para usar las características de EmbedIA en el microcontrolador, necesitas incluir la inicialización del modelo y ejecución de inferencia en tu código, usando las funciones proporcionadas:

* ```void model_init(void)```: Inicializa el modelo en lenguaje C, cargando los pesos y parámetros convertidos de tu modelo entrenado (TensorFlow/Keras o Scikit-learn).
* ```int model_predict(input, * results)```: Ejecuta la inferencia usando los datos de entrada pasados como parámetro. Esta función construye la arquitectura completa del modelo concatenando las salidas de las capas en el orden correcto. Devuelve el resultado de la predicción y pobla el arreglo de resultados con puntuaciones de confianza o valores de salida.

<strong>Ejemplo (Clasificación):</strong>
```c
// inicialización del modelo
model_init();

// inferencia del modelo
int prediction = model_predict(input, &results);

// 'prediction' contiene el ID de clase predicho
// 'results' contiene las puntuaciones de confianza para cada clase
```

<strong>Ejemplo (Regresión o Multi-salida):</strong>
```c
// inicialización del modelo
model_init();

// inferencia del modelo - para modelos de regresión o multi-salida
int status = model_predict(input, &results);

// 'results' contiene los valores predichos
```
