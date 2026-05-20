<div align="center">
  <hr>
  <img src="docs/assets/images/logo3.png" width=20%/>
  <h4><strong>EmbedIA 是一个用于在微控制器上开发应用程序的机器学习框架。</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.96.0-blue"/></a>
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

EmbedIA 是一个紧凑轻量的机器学习框架，专为在硬件资源有限的微控制器上部署模型而设计。它支持神经网络模型（使用 TensorFlow/Keras 训练）和机器学习算法（来自 Scikit-learn），能够在嵌入式系统上高效执行推理。该框架兼容 Arduino IDE 的 C 和 C++ 语言，并支持多种微控制器（MCU）。

## 目录 <A NAME="table-of-contents"></A>
* [工作流程](#workflow)
* [层与模型](#layers)
* [快速入门](#started)
* [在 C 语言中使用 EmbedIA](#inC)


## 工作流程 <A NAME="workflow"></A>
EmbedIA 支持两类机器学习模型：

### 神经网络（TensorFlow/Keras）
1. <strong>模型生成：</strong>选择架构，配置超参数，并准备训练数据。
2. <strong>训练：</strong>使用 Python 中的 TensorFlow/Keras 训练神经网络。
3. <strong>EmbedIA 导出：</strong>使用 EmbedIA 转换器将模型转换并导出为 C/C++。
4. <strong>部署：</strong>在目标微控制器平台上编译项目。
5. <strong>推理：</strong>在嵌入式设备上运行预测。

### 机器学习模型（Scikit-learn）
1. <strong>模型训练：</strong>使用 Scikit-learn 训练分类器或回归器。
2. <strong>EmbedIA 导出：</strong>使用 EmbedIA 转换器将训练好的模型转换为 C/C++。
3. <strong>部署：</strong>在目标微控制器平台上编译项目。
4. <strong>推理：</strong>在嵌入式设备上运行预测。

<p align="center"> <img src="docs/assets/images/workflow.png" width=90%/> </p>


## 层与模型 <A NAME="layers"></A>
EmbedIA 支持丰富的神经网络层集合以及主流机器学习框架的模型：

### 神经网络层（TensorFlow/Keras）

**卷积层：**
* <a href="https://keras.io/api/layers/convolution_layers/convolution1d/">Conv1D</a>
* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/depthwise_convolution2d/">DepthwiseConv2D</a>

**核心层：**
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>

**池化层：**
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling1d/">MaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling1d/">GlobalMaxPooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_max_pooling2d/">GlobalMaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling1d/">AveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/average_pooling2d/">AveragePooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling1d/">GlobalAveragePooling1D</a>
* <a href="https://keras.io/api/layers/pooling_layers/global_average_pooling2d/">GlobalAveragePooling2D</a>

**重塑层：**
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>
* <a href="https://keras.io/api/layers/reshaping_layers/zero_padding2d/">ZeroPadding2D</a>

**归一化层：**
* <a href="https://keras.io/api/layers/normalization_layers/batch_normalization/">BatchNormalization</a>

**激活函数：**
* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#leakyrelu-function">LeakyReLU</a>
* <a href="https://keras.io/api/layers/activations/#relu6-function">ReLU6</a>
* <a href="https://keras.io/api/layers/activations/#sigmoid-function">Sigmoid</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>
* <a href="https://keras.io/api/layers/activations/#softplus-function">Softplus</a>
* <a href="https://keras.io/api/layers/activations/#softsign-function">Softsign</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>

**量化层（Larq）：**
* <a href="https://docs.larq.dev/larq/api/layers/#quantconv2d">QuantConv2D</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantdense">QuantDense</a>
* <a href="https://docs.larq.dev/larq/api/layers/#quantseparableconv2d">QuantSeparableConv2D</a>

### 机器学习模型（Scikit-learn）

**预处理：**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html">MaxAbsScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html">RobustScaler</a>

**分类与回归模型：**
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">K 近邻分类器（K-Nearest Neighbors Classifier）</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">K 近邻回归器（K-Nearest Neighbors Regressor）</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">支持向量机（SVM）分类器</a>
* <a href="https://scikit-learn.org/stable/modules/svm.html">线性 SVM 分类器</a>
* <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">决策树分类器（Decision Tree Classifier）</a>

### 原生信号处理层

* **STFT：** 用于一维多频谱分析的短时傅里叶变换
* **频谱图（Spectrogram）：** 用于音频和信号分析的信号处理层

## 快速入门 <A NAME="started"></A>
要使用 EmbedIA Python 转换器，第一步是克隆仓库：

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

打开 <a href="https://github.com/Embed-ML/EmbedIA/blob/main/create_embedia_project.py">create_embedia_project.py</a> 脚本并配置转换器参数。该脚本同时支持 TensorFlow/Keras 模型和 Scikit-learn 模型：

**核心参数：**
* _OUTPUT_FOLDER_: 输出文件夹路径
* _PROJECT_NAME_: 生成的项目名称
* _MODEL_FILE_: 模型路径（Keras 使用 .h5 格式，Scikit-learn 使用序列化模型）

**配置选项：**
* _options.embedia_folder_: EmbedIA 文件所在文件夹：
  * ```options.embedia_folder = ...```
* _options.project_type_: 可用的项目类型：
  * ```ProjectType.ARDUINO```
  * ```ProjectType.C```
  * ```ProjectType.CPP```
  * ```ProjectType.CODEBLOCK```
  * ```ProjectType.CMAKE_C```
  * ```ProjectType.CMAKE_CPP```
* _options.micro_: 可用的微控制器类型选择：
  * ```ModelMicro.GENERIC```
  * ```ModelMicro.ESP32```
* _options.data_type_: 可用的数据类型选择：
  * ```ModelDataType.FLOAT```
  * ```ModelDataType.FIXED32```
  * ```ModelDataType.FIXED16```
  * ```ModelDataType.FIXED8```
  * ```ModelDataType.QUANT8```
  * ```ModelDataType.FULL_QUANT8```
  * ```ModelDataType.BINARY```
  * ```ModelDataType.BINARY_FIXED32```
  * ```ModelDataType.BINARY_FLOAT16```
* _options.fixed_precision_: 定点数据类型的小数位数（默认为 None）：
  * ```options.fixed_precision = 16```
* _options.tamano_bloque_: 二进制层块大小选项：
  * ```BinaryBlockSize.Bits8```
  * ```BinaryBlockSize.Bits16```
  * ```BinaryBlockSize.Bits32```
  * ```BinaryBlockSize.Bits64```
* _options.debug_mode_: 调试函数的包含与使用选项：
  * ```DebugMode.DISCARD```
  * ```DebugMode.DISABLED```
  * ```DebugMode.HEADERS```
  * ```DebugMode.DATA```
* _options.files_: 要执行的文件选择：
  * ```ProjectFiles.ALL()```
  * ```{ProjectFiles.MAIN}```
  * ```{ProjectFiles.MODEL}```
  * ```{ProjectFiles.LIBRARY}```
* _options.model_: 要转换的支持模型（TensorFlow/Keras、Scikit-Learn 等）
* _options.preprocessing_: 数据预处理的列表/对象（例如：归一化）
  * ```options.preprocessing_ = []```
* _options.example_data_: 作为示例包含的数据数组：
  * ```options.example_data = samples```
* _options.example_labels_: 示例的标签数组（用于分类）：
  * ```options.example_labels = labels```
* _options.baud_rate_: 仅限 Arduino，设置串口设备速度：
  * ```options.baud_rate = 9600```
* _options.verbose_: 项目生成期间的详细输出：
  * ```options.verbose = True```
* _options.clean_output_: 若为 True，则删除输出文件夹并重新开始导出：
  * ```options.clean_output = True```
* _options.output_subfolder_: 存储所有 EmbedIA 文件的文件夹名称：
  * ```options.output_subfolder = 'embedia'```

按如下方式运行脚本：
```bash
python create_embedia_project.py
```

如果过程成功，将显示一条消息，指示项目的生成位置。

**示例：**
* <strong>TensorFlow/Keras：</strong> 请查看 <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb">Google Colab 笔记本</a>，其中包含将在 MNIST 数据集上训练的 CNN 模型转换为 C 语言的完整示例。
* <strong>仿真：</strong> 在 <a href="https://wokwi.com/projects/359745013247499265">Wokwi 模拟器</a>中在线试用生成的代码。


## 在 C/C++ 中使用 EmbedIA <A NAME="inC"></A>
要在微控制器中使用 EmbedIA 的功能，需要在代码中使用提供的函数包含模型初始化和推理执行：

* ```void model_init(void)```: 用 C 语言初始化模型，加载从训练模型（TensorFlow/Keras 或 Scikit-learn）转换而来的权重和参数。
* ```int model_predict(input, * results)```: 使用作为参数传入的输入数据执行推理。该函数通过按正确顺序连接各层输出来构建完整的模型架构。返回预测结果，并将置信度分数或输出值填充到 results 数组中。

<strong>示例（分类）：</strong>
```c
// 模型初始化
model_init();

// 模型推理
int prediction = model_predict(input, &results);

// 'prediction' 包含预测的类别 ID
// 'results' 包含每个类别的置信度分数
```

<strong>示例（回归或多输出）：</strong>
```c
// 模型初始化
model_init();

// 模型推理 - 用于回归或多输出模型
int status = model_predict(input, &results);

// 'results' 包含预测值
```
