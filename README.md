<div align="center">
  <hr>
  <img src="images/logo3.png" width=20%/>
  <h4><strong>EmbedIA is a machine learning framework for developing applications on microcontrollers.</strong></h4>
  <a href="https://github.com/Embed-ML/EmbedIA"><img src="https://img.shields.io/badge/version-0.7.0-blue"/></a>  
  <a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
  <hr>
</div>

EmbedIA is a compact and lightweight framework capable of providing the necessary functionalities for the execution of inferences from Convolutional Neural Network models, created and trained using the Python Tensorflow/Keras library, on microcontrollers with limited hardware resources. It is designed to be compatible with the C and C++ languages for the Arduino IDE, both with support for a wide range of MCUs.

## Table of Contents <A NAME="tabla-de-contenidos"></A>
* [Workflow](#workflow)
* [Layers](#layers)
* [Getting started](#started)


## Workflow 🔨 <A NAME="workflow"></A>
For the conversion and use of Neural Network models in microcontrollers using embedia, the following workflow must be followed:

1. <strong>Generation of the model:</strong> Architecture selection, network hyperparameters and training data.
2. <strong>Training:</strong> Neural Network Training Using Tensorflow/Keras in Python.
3. <strong>EmbedIA Export:</strong> Export of C/C++ application with model and necessary libraries using the EmbedIA converter.
4. <strong>Solution Deployment:</strong> Project Compilation on the Microcontroller Platform.
5. <strong>Running Inferences:</strong>Running Inferences on the device.

<p align="center"> <img src="images/workflow.png" width=90%/> </p>


## Layers 🧅 <A NAME="layers"></A>
Currently it is possible to incorporate certain layers to the neural network model for execution on microcontrollers. The layers supported by EmbedIA, implemented in the C library, are the following:

* <a href="https://keras.io/api/layers/convolution_layers/convolution2d/">Conv2D</a>
* <a href="https://keras.io/api/layers/convolution_layers/separable_convolution2d/">SeparableConv2D</a>
* <a href="https://keras.io/api/layers/core_layers/dense/">Dense</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">MaxPooling2D</a>
* <a href="https://keras.io/api/layers/pooling_layers/max_pooling2d/">AveragePooling</a>
* <a href="https://keras.io/api/layers/reshaping_layers/flatten/">Flatten</a>

Activation functions are listed below:

* <a href="https://keras.io/api/layers/activations/#relu-function">ReLU</a>
* <a href="https://keras.io/api/layers/activations/#tanh-function">Tanh</a>
* <a href="https://keras.io/api/layers/activations/#softmax-function">Softmax</a>

## Getting started 🚀 <A NAME="started"></A>
In order to use the EmbedIA Python converter, the first step is to clone the repository

```bash
git clone https://github.com/Embed-ML/EmbedIA.git
cd EmbedIA
```

Open the create_embedia_project.py script and configure the converter parameters

...


In the following Colab there is an example of the use of the EmbedIA converter to create a project in C language for the classification of the images of the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">digits dataset</a>: 
<p align=center><a href="https://colab.research.google.com/github/Embed-ML/EmbedIA/blob/main/Using_EmbedIA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></p>
