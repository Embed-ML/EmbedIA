<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA — это фреймворк машинного обучения для разработки приложений на микроконтроллерах.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

**Языки:** [English](README.md) | [Español](README_ES.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Italiano](README_IT.md) | [Português](README_PT.md) | [Русский](README_RU.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

EmbedIA — это компактный и легкий фреймворк машинного обучения для развертывания моделей на микроконтроллерах с ограниченными аппаратными ресурсами. Он поддерживает как модели нейронных сетей (обученные с помощью TensorFlow/Keras), так и алгоритмы машинного обучения (из Scikit-learn), обеспечивая эффективное выполнение вывода на встраиваемых системах. Он разработан для совместимости с языками C и C++ для Arduino IDE и поддерживает широкий спектр микроконтроллеров (MCU).

## 📑 Содержание <A NAME="tabla-de-contenidos"></A>
* [Рабочий процесс](#workflow)
* [Слои](#layers)
* [Начало работы](#started)
* [EmbedIA на C](#inC)


## 🔨 Рабочий процесс <A NAME="workflow"></A>
EmbedIA поддерживает два типа моделей машинного обучения:

### 🧠 Для нейронных сетей (TensorFlow/Keras)
1. <strong>Создание модели:</strong> Выбрать архитектуру, настроить гиперпараметры и подготовить обучающие данные.
2. <strong>Обучение:</strong> Обучить нейронную сеть с использованием TensorFlow/Keras на Python.
3. <strong>Экспорт EmbedIA:</strong> Преобразовать и экспортировать модель в C/C++ с помощью конвертера EmbedIA.
4. <strong>Развертывание:</strong> Скомпилировать проект на целевой платформе микроконтроллера.
5. <strong>Вывод:</strong> Выполнить предсказания на встраиваемом устройстве.

### 🤖 Для моделей машинного обучения (Scikit-learn)
1. <strong>Обучение модели:</strong> Обучить классификаторы или регрессоры с использованием Scikit-learn.
2. <strong>Экспорт EmbedIA:</strong> Преобразовать обученную модель в C/C++ с помощью конвертера EmbedIA.
3. <strong>Развертывание:</strong> Скомпилировать проект на целевой платформе микроконтроллера.
4. <strong>Вывод:</strong> Выполнить предсказания на встраиваемом устройстве.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 🧅 Слои и модели <A NAME="layers"></A>
EmbedIA поддерживает полный набор слоев для нейронных сетей и моделей из популярных фреймворков машинного обучения:

### ⚡ Слои нейронных сетей (TensorFlow/Keras)

**Сверточные слои:**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Основные слои:**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Слои пулинга:**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Слои изменения формы:**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Слои нормализации:**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Функции активации:**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Квантованные слои (Larq):**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 🎯 Модели машинного обучения (Scikit-learn)

**Предобработка:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Модели классификации и регрессии:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Классификатор K-ближайших соседей</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">Регрессор K-ближайших соседей</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Классификатор метода опорных векторов (SVM)</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Линейный классификатор SVM</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Классификатор дерева решений</a>

### 🔊 Нативные слои обработки сигналов

* **STFT:** Кратковременное преобразование Фурье для многоспектрального анализа 1D
* **Спектрограмма:** Слой обработки сигналов для анализа аудио и сигналов

## 🚀 Начало работы <A NAME="started"></A>
Чтобы использовать конвертер Python EmbedIA, первым шагом является клонирование репозитория:

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Откройте скрипт <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> и настройте параметры конвертера. Этот скрипт поддерживает как модели TensorFlow/Keras, так и модели Scikit-learn:

**Основные параметры:**
* _OUTPUT_FOLDER_: путь к выходной папке
* _PROJECT_NAME_: имя сгенерированного проекта
* _MODEL_FILE_: путь к модели (формат .h5 для Keras или сериализованная модель для Scikit-learn)

**Параметры конфигурации:**
* _options.embedia_folder_: папка файлов EmbedIA:
  * ```options.embedia_folder = ...```
* _options.project_type_: тип проекта среди доступных:
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: выбор типа микроконтроллера среди доступных:
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: выбор типа данных среди доступных:
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: количество дробных битов для типов данных с фиксированной точкой (None по умолчанию):
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: параметры размера блока для двоичных слоев:
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: параметры включения и использования функций отладки:
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: Выбор файлов для выполнения:
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: поддерживаемая модель для преобразования (TensorFlow/Keras, Scikit-Learn и т.д.)
* _options.preprocessing_: список/объект для предобработки данных (например: нормализация)
  * ```options.preprocessing_ = []```
* _options.example_data_: массив данных для включения в качестве примеров:
  * ```options.example_data = samples```
* _options.example_labels_: массив меток для примеров (классификация):
  * ```options.example_labels = labels```
* _options.baud_rate_: Только для Arduino, установить скорость последовательного устройства:
  * ```options.baud_rate = 9600```
* _options.verbose_: подробный вывод во время генерации проекта:
  * ```options.verbose = True```
* _options.clean_output_: если True, удалить выходную папку и начать чистый экспорт:
  * ```options.clean_output = True```
* _options.output_subfolder_: имя папки для хранения всех файлов embedia:
  * ```options.output_subfolder = 'embedia'```

Запустите скрипт следующим образом:
```bash
python create_embedia_project.py
```

Если процесс прошел успешно, будет отображено сообщение, указывающее, где был сгенерирован проект.

**Примеры:**
* <strong>TensorFlow/Keras:</strong> Ознакомьтесь с <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">блокнотом Google Colab</a> для полного примера преобразования модели CNN, обученной на наборе данных MNIST, в язык C.
* <strong>Симуляция:</strong> Попробуйте сгенерированный код онлайн в <a href="https://wokwi.com/projects/359745013247499265">симуляторе Wokwi</a>.


## 👍 EmbedIA на C/C++ <A NAME="inC"></A>
Чтобы использовать функции EmbedIA в микроконтроллере, вам необходимо включить инициализацию модели и выполнение вывода в ваш код, используя предоставленные функции:

* ```void model_init(void)```: Инициализирует модель на языке C, загружая веса и параметры, преобразованные из вашей обученной модели (TensorFlow/Keras или Scikit-learn).
* ```int model_predict(input, * results)```: Выполняет вывод, используя входные данные, переданные в качестве параметра. Эта функция строит полную архитектуру модели, объединяя выходы слоев в правильном порядке. Она возвращает результат предсказания и заполняет массив результатов оценками достоверности или выходными значениями.

<strong>Пример (Классификация):</strong>
```c
// инициализация модели
model_init();

// вывод модели
int prediction = model_predict(input, &results);

// 'prediction' содержит ID предсказанного класса
// 'results' содержит оценки достоверности для каждого класса
```

<strong>Пример (Регрессия или множественный вывод):</strong>
```c
// инициализация модели
model_init();

// вывод модели - для моделей регрессии или множественного вывода
int status = model_predict(input, &results);

// 'results' содержит предсказанные значения
```
