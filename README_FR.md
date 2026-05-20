<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA est un framework de machine learning pour développer des applications sur microcontrôleurs.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

**Langues:** [English](README.md) | [Español](README_ES.md) | [Deutsch](README_DE.md) | [Français](README_FR.md) | [Italiano](README_IT.md) | [Português](README_PT.md) | [Русский](README_RU.md) | [中文](README_ZH.md) | [日本語](README_JA.md)

EmbedIA est un framework de machine learning compact et léger pour déployer des modèles sur des microcontrôleurs avec des ressources matérielles limitées. Il prend en charge à la fois les modèles de réseaux neuronaux (entraînés avec TensorFlow/Keras) et les algorithmes de machine learning (de Scikit-learn), permettant une exécution efficace de l'inférence sur les systèmes embarqués. Il est conçu pour être compatible avec les langages C et C++ pour l'IDE Arduino et prend en charge une large gamme de microcontrôleurs (MCU).

## 📑 Table des matières <A NAME="tabla-de-contenidos"></A>
* [Flux de travail](#workflow)
* [Couches](#layers)
* [Premiers pas](#started)
* [EmbedIA en C](#inC)


## 🔨 Flux de travail <A NAME="workflow"></A>
EmbedIA prend en charge deux types de modèles de machine learning :

### 🧠 Pour les réseaux neuronaux (TensorFlow/Keras)
1. <strong>Génération du modèle :</strong> Sélectionner l'architecture, configurer les hyperparamètres et préparer les données d'entraînement.
2. <strong>Entraînement :</strong> Entraîner votre réseau neuronal en utilisant TensorFlow/Keras en Python.
3. <strong>Export EmbedIA :</strong> Convertir et exporter le modèle en C/C++ en utilisant le convertisseur EmbedIA.
4. <strong>Déploiement :</strong> Compiler le projet sur votre plateforme de microcontrôleur cible.
5. <strong>Inférence :</strong> Exécuter des prédictions sur le dispositif embarqué.

### 🤖 Pour les modèles de machine learning (Scikit-learn)
1. <strong>Entraînement du modèle :</strong> Entraîner des classificateurs ou des régresseurs en utilisant Scikit-learn.
2. <strong>Export EmbedIA :</strong> Convertir le modèle entraîné en C/C++ en utilisant le convertisseur EmbedIA.
3. <strong>Déploiement :</strong> Compiler le projet sur votre plateforme de microcontrôleur cible.
4. <strong>Inférence :</strong> Exécuter des prédictions sur le dispositif embarqué.

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 🧅 Couches et modèles <A NAME="layers"></A>
EmbedIA prend en charge un ensemble complet de couches pour les réseaux neuronaux et les modèles des frameworks de machine learning populaires :

### ⚡ Couches de réseaux neuronaux (TensorFlow/Keras)

**Couches convolutionnelles :**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**Couches principales :**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**Couches de pooling :**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**Couches de remodelage :**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**Couches de normalisation :**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**Fonctions d'activation :**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**Couches quantifiées (Larq) :**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 🎯 Modèles de machine learning (Scikit-learn)

**Prétraitement :**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**Modèles de classification et de régression :**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">Classificateur K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">Régresseur K-Nearest Neighbors</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">Classificateur Support Vector Machine (SVM)</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">Classificateur SVM linéaire</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Classificateur d'arbre de décision</a>

### 🔊 Couches natives de traitement du signal

* **STFT :** Transformée de Fourier à court terme pour l'analyse multi-spectrale 1D
* **Spectrogramme :** Couche de traitement du signal pour l'analyse audio et des signaux

## 🚀 Premiers pas <A NAME="started"></A>
Pour utiliser le convertisseur Python EmbedIA, la première étape consiste à cloner le dépôt :

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Ouvrez le script <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> et configurez les paramètres du convertisseur. Ce script prend en charge à la fois les modèles TensorFlow/Keras et les modèles Scikit-learn :

**Paramètres principaux :**
* _OUTPUT_FOLDER_ : chemin du dossier de sortie
* _PROJECT_NAME_ : nom du projet généré
* _MODEL_FILE_ : chemin du modèle (format .h5 pour Keras, ou modèle sérialisé pour Scikit-learn)

**Options de configuration :**
* _options.embedia_folder_ : dossier des fichiers EmbedIA :
  * ```options.embedia_folder = ...```
* _options.project_type_ : type de projet parmi ceux disponibles :
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_ : sélection du type de microcontrôleur parmi ceux disponibles :
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_ : sélection du type de données parmi ceux disponibles :
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_ : nombre de bits fractionnaires pour les types de données à virgule fixe (None par défaut) :
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_ : options pour la taille de bloc des couches binaires :
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_ : options pour l'inclusion et l'utilisation des fonctions de débogage :
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_ : Sélection des fichiers à exécuter :
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_ : modèle pris en charge à convertir (TensorFlow/Keras, Scikit-Learn, etc.)
* _options.preprocessing_ : liste/objet pour le prétraitement des données (par ex. : normalisation)
  * ```options.preprocessing_ = []```
* _options.example_data_ : tableau de données à inclure comme exemples :
  * ```options.example_data = samples```
* _options.example_labels_ : tableau d'étiquettes pour les exemples (classification) :
  * ```options.example_labels = labels```
* _options.baud_rate_ : Arduino uniquement, définir la vitesse du périphérique série :
  * ```options.baud_rate = 9600```
* _options.verbose_ : sortie détaillée pendant la génération du projet :
  * ```options.verbose = True```
* _options.clean_output_ : si True, supprimer le dossier de sortie et démarrer une exportation propre :
  * ```options.clean_output = True```
* _options.output_subfolder_ : nom du dossier pour stocker tous les fichiers embedia :
  * ```options.output_subfolder = 'embedia'```

Exécutez le script comme suit :
```bash
python create_embedia_project.py
```

Si le processus a réussi, un message s'affichera indiquant où le projet a été généré.

**Exemples :**
* <strong>TensorFlow/Keras :</strong> Consultez le <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">notebook Google Colab</a> pour un exemple complet de conversion d'un modèle CNN entraîné sur le jeu de données MNIST en langage C.
* <strong>Simulation :</strong> Essayez le code généré en ligne dans le <a href="https://wokwi.com/projects/359745013247499265">simulateur Wokwi</a>.


## 👍 EmbedIA en C/C++ <A NAME="inC"></A>
Pour utiliser les fonctionnalités d'EmbedIA dans le microcontrôleur, vous devez inclure l'initialisation du modèle et l'exécution de l'inférence dans votre code, en utilisant les fonctions fournies :

* ```void model_init(void)``` : Initialise le modèle en langage C, en chargeant les poids et les paramètres convertis à partir de votre modèle entraîné (TensorFlow/Keras ou Scikit-learn).
* ```int model_predict(input, * results)``` : Exécute l'inférence en utilisant les données d'entrée passées en paramètre. Cette fonction construit l'architecture complète du modèle en concaténant les sorties des couches dans le bon ordre. Elle renvoie le résultat de la prédiction et remplit le tableau de résultats avec les scores de confiance ou les valeurs de sortie.

<strong>Exemple (Classification) :</strong>
```c
// initialisation du modèle
model_init();

// inférence du modèle
int prediction = model_predict(input, &results);

// 'prediction' contient l'ID de classe prédit
// 'results' contient les scores de confiance pour chaque classe
```

<strong>Exemple (Régression ou multi-sortie) :</strong>
```c
// initialisation du modèle
model_init();

// inférence du modèle - pour les modèles de régression ou multi-sortie
int status = model_predict(input, &results);

// 'results' contient les valeurs prédites
```
