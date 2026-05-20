<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA ist ein Machine-Learning-Framework für die Entwicklung von Anwendungen auf Mikrocontrollern.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

**Sprachen:** [English](README.md) | [Español](README_ES.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Italiano](README_IT.md) | [Português](README_PT.md) | [Русский](README_RU.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

EmbedIA ist ein kompaktes und leichtgewichtiges Machine-Learning-Framework für die Bereitstellung von Modellen auf Mikrocontrollern mit begrenzten Hardwareressourcen. Es unterstützt sowohl neuronale Netzwerkmodelle (trainiert mit TensorFlow/Keras) als auch Machine-Learning-Algorithmen (von Scikit-learn) und ermöglicht eine effiziente Inferenzausführung auf eingebetteten Systemen. Es ist so konzipiert, dass es mit den Sprachen C und C++ für die Arduino IDE kompatibel ist und eine breite Palette von Mikrocontrollern (MCUs) unterstützt.

## 📑 Inhaltsverzeichnis <A NAME="tabla-de-contenidos"></A>
* [Arbeitsablauf](#workflow)
* [Schichten](#layers)
* [Erste Schritte](#started)
* [EmbedIA in C](#inC)


## 🔨 Arbeitsablauf <A NAME="workflow"></A>
EmbedIA unterstützt zwei Arten von Machine-Learning-Modellen:

### 🧠 Für neuronale Netzwerke (TensorFlow/Keras)
1. <strong>Modellerstellung:</strong> Architektur auswählen, Hyperparameter konfigurieren und Trainingsdaten vorbereiten.
2. <strong>Training:</strong> Trainieren Sie Ihr neuronales Netzwerk mit TensorFlow/Keras in Python.
3. <strong>EmbedIA-Export:</strong> Konvertieren und exportieren Sie das Modell nach C/C++ mit dem EmbedIA-Konverter.
4. <strong>Bereitstellung:</strong> Kompilieren Sie das Projekt auf Ihrer Ziel-Mikrocontroller-Plattform.
5. <strong>Inferenz:</strong> Führen Sie Vorhersagen auf dem eingebetteten Gerät aus.

### 🤖 Für Machine-Learning-Modelle (Scikit-learn)
1. <strong>Modelltraining:</strong> Trainieren Sie Klassifikatoren oder Regressoren mit Scikit-learn.
2. <strong>EmbedIA-Export:</strong> Konvertieren Sie das trainierte Modell nach C/C++ mit dem EmbedIA-Konverter.
3. <strong>Bereitstellung:</strong> Kompilieren Sie das Projekt auf Ihrer Ziel-Mikrocontroller-Plattform.
4. <strong>Inferenz:</strong> Führen Sie Vorhersagen auf dem eingebetteten Gerät aus.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 🧅 Schichten und Modelle <A NAME="layers"></A>
EmbedIA unterstützt einen umfassenden Satz von Schichten für neuronale Netzwerke und Modelle aus beliebten Machine-Learning-Frameworks:

### ⚡ Neuronale Netzwerkschichten (TensorFlow/Keras)

**Faltungsschichten:**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Kernschichten:**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Pooling-Schichten:**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Umformungsschichten:**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Normalisierungsschichten:**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Aktivierungsfunktionen:**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Quantisierte Schichten (Larq):**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 🎯 Machine-Learning-Modelle (Scikit-learn)

**Vorverarbeitung:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Klassifizierungs- und Regressionsmodelle:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K-Nearest Neighbors Klassifikator</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K-Nearest Neighbors Regressor</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Support Vector Machine (SVM) Klassifikator</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Linearer SVM-Klassifikator</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Entscheidungsbaum-Klassifikator</a>

### 🔊 Native Signalverarbeitungsschichten

* **STFT:** Kurzzeit-Fourier-Transformation für 1D-Multispektralanalyse
* **Spektrogramm:** Signalverarbeitungsschicht für Audio- und Signalanalyse

## 🚀 Erste Schritte <A NAME="started"></A>
Um den EmbedIA Python-Konverter zu verwenden, besteht der erste Schritt darin, das Repository zu klonen:

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Öffnen Sie das Skript <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> und konfigurieren Sie die Konverterparameter. Dieses Skript unterstützt sowohl TensorFlow/Keras-Modelle als auch Scikit-learn-Modelle:

**Hauptparameter:**
* _OUTPUT_FOLDER_: Ausgabeordnerpfad
* _PROJECT_NAME_: Name des generierten Projekts
* _MODEL_FILE_: Modellpfad (.h5-Format für Keras oder serialisiertes Modell für Scikit-learn)

**Konfigurationsoptionen:**
* _options.embedia_folder_: Ordner der EmbedIA-Dateien:
  * ```options.embedia_folder = ...```
* _options.project_type_: Projekttyp unter den verfügbaren:
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: Auswahl des Mikrocontroller-Typs unter den verfügbaren:
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: Auswahl des Datentyps unter den verfügbaren:
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: Anzahl der Bruchbits für Festkomma-Datentypen (None für Standard):
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: Optionen für die Blockgröße binärer Schichten:
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: Optionen für die Einbeziehung und Verwendung von Debug-Funktionen:
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: Auswahl der auszuführenden Dateien:
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: Unterstütztes Modell zum Konvertieren (TensorFlow/Keras, Scikit-Learn usw.)
* _options.preprocessing_: Liste/Objekt für die Datenvorverarbeitung (z.B.: Normalisierung)
  * ```options.preprocessing_ = []```
* _options.example_data_: Array von Daten, die als Beispiele einbezogen werden sollen:
  * ```options.example_data = samples```
* _options.example_labels_: Array von Labels für Beispiele (Klassifizierung):
  * ```options.example_labels = labels```
* _options.baud_rate_: Nur für Arduino, Geschwindigkeit des seriellen Geräts einstellen:
  * ```options.baud_rate = 9600```
* _options.verbose_: Ausführliche Ausgabe während der Projektgenerierung:
  * ```options.verbose = True```
* _options.clean_output_: Wenn True, Ausgabeordner entfernen und einen sauberen Export starten:
  * ```options.clean_output = True```
* _options.output_subfolder_: Name des Ordners zum Speichern aller embedia-Dateien:
  * ```options.output_subfolder = 'embedia'```

Führen Sie das Skript wie folgt aus:
```bash
python create_embedia_project.py
```

Wenn der Prozess erfolgreich war, wird eine Meldung angezeigt, die angibt, wo das Projekt generiert wurde.

**Beispiele:**
* <strong>TensorFlow/Keras:</strong> Sehen Sie sich das <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">Google Colab-Notebook</a> für ein vollständiges Beispiel der Konvertierung eines auf dem MNIST-Datensatz trainierten CNN-Modells in die Sprache C an.
* <strong>Simulation:</strong> Probieren Sie den generierten Code online im <a href="https://wokwi.com/projects/359745013247499265">Wokwi-Simulator</a> aus.


## 👍 EmbedIA in C/C++ <A NAME="inC"></A>
Um die EmbedIA-Funktionen im Mikrocontroller zu verwenden, müssen Sie die Modellinitialisierung und Inferenzausführung in Ihren Code einbeziehen, indem Sie die bereitgestellten Funktionen verwenden:

* ```void model_init(void)```: Initialisiert das Modell in der Sprache C und lädt die Gewichte und Parameter, die aus Ihrem trainierten Modell (TensorFlow/Keras oder Scikit-learn) konvertiert wurden.
* ```int model_predict(input, * results)```: Führt die Inferenz mit den als Parameter übergebenen Eingabedaten aus. Diese Funktion erstellt die vollständige Modellarchitektur, indem sie die Schichtausgaben in der richtigen Reihenfolge verkettet. Sie gibt das Vorhersageergebnis zurück und füllt das Ergebnis-Array mit Konfidenzwerten oder Ausgabewerten.

<strong>Beispiel (Klassifizierung):</strong>
```c
// Modellinitialisierung
model_init();

// Modellinferenz
int prediction = model_predict(input, &results);

// 'prediction' enthält die vorhergesagte Klassen-ID
// 'results' enthält die Konfidenzwerte für jede Klasse
```

<strong>Beispiel (Regression oder Multi-Output):</strong>
```c
// Modellinitialisierung
model_init();

// Modellinferenz - für Regressions- oder Multi-Output-Modelle
int status = model_predict(input, &results);

// 'results' enthält die vorhergesagten Werte
```
