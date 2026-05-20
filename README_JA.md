<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA は、マイクロコントローラー向けアプリケーション開発のための機械学習フレームワークです。</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

EmbedIA は、ハードウェアリソースが限られたマイクロコントローラーへのモデルデプロイ向けに設計された、コンパクトで軽量な機械学習フレームワークです。TensorFlow/Keras で学習したニューラルネットワークモデルと、Scikit-learn の機械学習アルゴリズムの両方をサポートし、組み込みシステム上での効率的な推論実行を可能にします。Arduino IDE 向けの C および C++ 言語と互換性があり、幅広いマイクロコントローラー（MCU）をサポートしています。

## 目次 <A NAME="table-of-contents"></A>
* [ワークフロー](#workflow)
* [レイヤー](#layers)
* [はじめに](#started)
* [C言語での EmbedIA](#inC)


## ワークフロー <A NAME="workflow"></A>
EmbedIA は 2 種類の機械学習モデルをサポートしています：

### ニューラルネットワーク（TensorFlow/Keras）
1. <strong>モデル生成：</strong>アーキテクチャを選択し、ハイパーパラメータを設定して学習データを準備します。
2. <strong>学習：</strong>Python の TensorFlow/Keras を使用してニューラルネットワークを学習します。
3. <strong>EmbedIA エクスポート：</strong>EmbedIA コンバーターを使用してモデルを C/C++ に変換・エクスポートします。
4. <strong>デプロイ：</strong>対象のマイクロコントローラープラットフォーム上でプロジェクトをコンパイルします。
5. <strong>推論：</strong>組み込みデバイス上で予測を実行します。

### 機械学習モデル（Scikit-learn）
1. <strong>モデル学習：</strong>Scikit-learn を使用して分類器または回帰器を学習します。
2. <strong>EmbedIA エクスポート：</strong>EmbedIA コンバーターを使用して学習済みモデルを C/C++ に変換します。
3. <strong>デプロイ：</strong>対象のマイクロコントローラープラットフォーム上でプロジェクトをコンパイルします。
4. <strong>推論：</strong>組み込みデバイス上で予測を実行します。

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## レイヤーとモデル <A NAME="layers"></A>
EmbedIA は、ニューラルネットワーク向けの包括的なレイヤーセットと、一般的な機械学習フレームワークのモデルをサポートしています：

### ニューラルネットワークレイヤー（TensorFlow/Keras）

**畳み込みレイヤー：**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**コアレイヤー：**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**プーリングレイヤー：**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**リシェイプレイヤー：**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**正規化レイヤー：**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**活性化関数：**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**量子化レイヤー（Larq）：**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 機械学習モデル（Scikit-learn）

**前処理：**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**分類・回帰モデル：**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K近傍法分類器（K-Nearest Neighbors Classifier）</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K近傍法回帰器（K-Nearest Neighbors Regressor）</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">サポートベクターマシン（SVM）分類器</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">線形 SVM 分類器</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">決定木分類器（Decision Tree Classifier）</a>

### ネイティブ信号処理レイヤー

* **STFT：** 1次元マルチスペクトル解析のための短時間フーリエ変換
* **スペクトログラム：** 音声・信号解析のための信号処理レイヤー

## はじめに <A NAME="started"></A>
EmbedIA Python コンバーターを使用するには、まずリポジトリをクローンします：

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

<a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> スクリプトを開き、コンバーターのパラメータを設定します。このスクリプトは TensorFlow/Keras モデルと Scikit-learn モデルの両方をサポートしています：

**主要パラメータ：**
* _OUTPUT_FOLDER_: 出力フォルダのパス
* _PROJECT_NAME_: 生成プロジェクト名
* _MODEL_FILE_: モデルのパス（Keras は .h5 形式、Scikit-learn はピクルル化されたモデル）

**設定オプション：**
* _options.embedia_folder_: EmbedIA ファイルのフォルダ：
  * ```options.embedia_folder = ...```
* _options.project_type_: 利用可能なプロジェクトタイプ：
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: 利用可能なマイクロコントローラータイプの選択：
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: 利用可能なデータ型の選択：
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: 固定小数点データ型の小数部ビット数（デフォルトの場合は None）：
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: バイナリレイヤーのブロックサイズのオプション：
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: デバッグ関数の組み込みと使用に関するオプション：
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: 実行するファイルの選択：
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: 変換対象のサポートモデル（TensorFlow/Keras、Scikit-Learn など）
* _options.preprocessing_: データ前処理のリスト/オブジェクト（例：正規化）
  * ```options.preprocessing_ = []```
* _options.example_data_: サンプルとして含めるデータの配列：
  * ```options.example_data = samples```
* _options.example_labels_: サンプルのラベルの配列（分類用）：
  * ```options.example_labels = labels```
* _options.baud_rate_: Arduino のみ、シリアルデバイスの速度を設定：
  * ```options.baud_rate = 9600```
* _options.verbose_: プロジェクト生成中の詳細出力：
  * ```options.verbose = True```
* _options.clean_output_: True の場合、出力フォルダを削除してクリーンなエクスポートを開始：
  * ```options.clean_output = True```
* _options.output_subfolder_: すべての EmbedIA ファイルを格納するフォルダ名：
  * ```options.output_subfolder = 'embedia'```

スクリプトを以下のように実行します：
```bash
python create_embedia_project.py
```

処理が正常に完了すると、プロジェクトが生成された場所を示すメッセージが表示されます。

**サンプル：**
* <strong>TensorFlow/Keras：</strong> MNIST データセットで学習した CNN モデルを C 言語に変換する完全なサンプルは、<a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">Google Colab ノートブック</a>をご覧ください。
* <strong>シミュレーション：</strong> <a href="https://wokwi.com/projects/359745013247499265">Wokwi シミュレーター</a>で生成されたコードをオンラインで試すことができます。


## C/C++ での EmbedIA <A NAME="inC"></A>
マイクロコントローラーで EmbedIA の機能を使用するには、提供されている関数を使ってコードにモデルの初期化と推論実行を組み込む必要があります：

* ```void model_init(void)```: C 言語でモデルを初期化し、学習済みモデル（TensorFlow/Keras または Scikit-learn）から変換された重みとパラメータを読み込みます。
* ```int model_predict(input, * results)```: パラメータとして渡された入力データを使用して推論を実行します。この関数は、レイヤーの出力を正しい順序で連結することでモデルアーキテクチャ全体を構築します。予測結果を返し、results 配列に信頼スコアまたは出力値を格納します。

<strong>サンプル（分類）：</strong>
```c
// モデルの初期化
model_init();

// モデル推論
int prediction = model_predict(input, &results);

// 'prediction' には予測されたクラス ID が格納されます
// 'results' には各クラスの信頼スコアが格納されます
```

<strong>サンプル（回帰またはマルチ出力）：</strong>
```c
// モデルの初期化
model_init();

// モデル推論 - 回帰またはマルチ出力モデルの場合
int status = model_predict(input, &results);

// 'results' には予測値が格納されます
```
