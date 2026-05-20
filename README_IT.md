<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA è un framework di machine learning per lo sviluppo di applicazioni su microcontrollori.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

**Lingue:** [English](README.md) | [Español](README_ES.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Italiano](README_IT.md) | [Português](README_PT.md) | [Русский](README_RU.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

EmbedIA è un framework di machine learning compatto e leggero per il deployment di modelli su microcontrollori con risorse hardware limitate. Supporta sia modelli di reti neurali (addestrati con TensorFlow/Keras) che algoritmi di machine learning (da Scikit-learn), consentendo un'esecuzione efficiente dell'inferenza su sistemi embedded. È progettato per essere compatibile con i linguaggi C e C++ per l'IDE Arduino e supporta un'ampia gamma di microcontrollori (MCU).

## 📑 Indice <A NAME="tabla-de-contenidos"></A>
* [Flusso di lavoro](#workflow)
* [Layer](#layers)
* [Primi passi](#started)
* [EmbedIA in C](#inC)


## 🔨 Flusso di lavoro <A NAME="workflow"></A>
EmbedIA supporta due tipi di modelli di machine learning:

### 🧠 Per Reti Neurali (TensorFlow/Keras)
1. <strong>Generazione del modello:</strong> Selezionare l'architettura, configurare gli iperparametri e preparare i dati di addestramento.
2. <strong>Addestramento:</strong> Addestrare la rete neurale utilizzando TensorFlow/Keras in Python.
3. <strong>Esportazione EmbedIA:</strong> Convertire ed esportare il modello in C/C++ utilizzando il convertitore EmbedIA.
4. <strong>Deployment:</strong> Compilare il progetto sulla piattaforma microcontrollore di destinazione.
5. <strong>Inferenza:</strong> Eseguire previsioni sul dispositivo embedded.

### 🤖 Per Modelli di Machine Learning (Scikit-learn)
1. <strong>Addestramento del modello:</strong> Addestrare classificatori o regressori utilizzando Scikit-learn.
2. <strong>Esportazione EmbedIA:</strong> Convertire il modello addestrato in C/C++ utilizzando il convertitore EmbedIA.
3. <strong>Deployment:</strong> Compilare il progetto sulla piattaforma microcontrollore di destinazione.
4. <strong>Inferenza:</strong> Eseguire previsioni sul dispositivo embedded.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 🧅 Layer e Modelli <A NAME="layers"></A>
EmbedIA supporta un set completo di layer per reti neurali e modelli dai framework di machine learning più popolari:

### ⚡ Layer di Reti Neurali (TensorFlow/Keras)

**Layer Convoluzionali:**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Layer Principali:**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Layer di Pooling:**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Layer di Reshaping:**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Layer di Normalizzazione:**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Funzioni di Attivazione:**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Layer Quantizzati (Larq):**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 🎯 Modelli di Machine Learning (Scikit-learn)

**Preprocessing:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Modelli di Classificazione e Regressione:**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Classificatore K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">Regressore K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Classificatore Support Vector Machine (SVM)</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Classificatore SVM Lineare</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Classificatore Decision Tree</a>

### 🔊 Layer Nativi di Elaborazione del Segnale

* **STFT:** Trasformata di Fourier a Tempo Breve per analisi multi-spettrale 1D
* **Spettrogramma:** Layer di elaborazione del segnale per analisi audio e segnali

## 🚀 Primi passi <A NAME="started"></A>
Per utilizzare il convertitore Python di EmbedIA, il primo passo è clonare il repository:

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Aprire lo script <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> e configurare i parametri del convertitore. Questo script supporta sia modelli TensorFlow/Keras che modelli Scikit-learn:

**Parametri Principali:**
* _OUTPUT_FOLDER_: percorso della cartella di output
* _PROJECT_NAME_: nome del progetto generato
* _MODEL_FILE_: percorso del modello (formato .h5 per Keras, o modello serializzato per Scikit-learn)

**Opzioni di Configurazione:**
* _options.embedia_folder_: cartella dei file EmbedIA:
  * ```options.embedia_folder = ...```
* _options.project_type_: tipo di progetto tra quelli disponibili:
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: selezione del tipo di microcontrollore tra quelli disponibili:
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: selezione del tipo di dato tra quelli disponibili:
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: numero di bit frazionari per tipi di dati a virgola fissa (None per predefinito):
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: opzioni per la dimensione del blocco dei layer binari:
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: opzioni per l'inclusione e l'uso delle funzioni di debug:
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: Selezione dei file da eseguire:
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: modello supportato da convertire (TensorFlow/Keras, Scikit-Learn, ecc.)
* _options.preprocessing_: lista/oggetto per il preprocessing dei dati (es: normalizzazione)
  * ```options.preprocessing_ = []```
* _options.example_data_: array di dati da includere come esempi:
  * ```options.example_data = samples```
* _options.example_labels_: array di etichette per gli esempi (classificazione):
  * ```options.example_labels = labels```
* _options.baud_rate_: Solo per Arduino, impostare la velocità del dispositivo Serial:
  * ```options.baud_rate = 9600```
* _options.verbose_: output dettagliato durante la generazione del progetto:
  * ```options.verbose = True```
* _options.clean_output_: se True, rimuove la cartella di output e avvia un'esportazione pulita:
  * ```options.clean_output = True```
* _options.output_subfolder_: nome della cartella per memorizzare tutti i file embedia:
  * ```options.output_subfolder = 'embedia'```

Eseguire lo script come segue:
```bash
python create_embedia_project.py
```

Se il processo ha avuto successo, verrà visualizzato un messaggio che indica dove è stato generato il progetto.

**Esempi:**
* <strong>TensorFlow/Keras:</strong> Consultare il <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">notebook di Google Colab</a> per un esempio completo di conversione di un modello CNN addestrato sul dataset MNIST in linguaggio C.
* <strong>Simulazione:</strong> Provare il codice generato online nel <a href="https://wokwi.com/projects/359745013247499265">simulatore Wokwi</a>.


## 👍 EmbedIA in C/C++ <A NAME="inC"></A>
Per utilizzare le funzionalità di EmbedIA nel microcontrollore, è necessario includere l'inizializzazione del modello e l'esecuzione dell'inferenza nel codice, utilizzando le funzioni fornite:

* ```void model_init(void)```: Inizializza il modello in linguaggio C, caricando i pesi e i parametri convertiti dal modello addestrato (TensorFlow/Keras o Scikit-learn).
* ```int model_predict(input, * results)```: Esegue l'inferenza utilizzando i dati di input passati come parametro. Questa funzione costruisce l'architettura completa del modello concatenando gli output dei layer nell'ordine corretto. Restituisce il risultato della previsione e popola l'array dei risultati con i punteggi di confidenza o i valori di output.

<strong>Esempio (Classificazione):</strong>
```c
// inizializzazione del modello
model_init();

// inferenza del modello
int prediction = model_predict(input, &results);

// 'prediction' contiene l'ID della classe prevista
// 'results' contiene i punteggi di confidenza per ogni classe
```

<strong>Esempio (Regressione o Multi-output):</strong>
```c
// inizializzazione del modello
model_init();

// inferenza del modello - per modelli di regressione o multi-output
int status = model_predict(input, &results);

// 'results' contiene i valori previsti
```
