# Changelog

## v0.98.0 - 2026-05-19

### Added

* New Layer: Logistic Regression
* Logistic regression layer with full scikit-learn support
* Wrapper at `wrappers/sklearn/logistic_regression.py`
* C implementation in all variants (float, fixed8, fixed16, quant8)
* New Conv1D and GlobalPooling support
  * Support for Conv1D layers
  * GlobalPooling (Max, Avg) for 1D/2D/3D
  * New C functions for 1D operations
* ProjectOptions extensions
  * `output_subfolder`: configurable subfolder for EmbedIA export files
  * `project_type`: project type with CMake export support
* C code generator improvements
  * Improved handling of buffers and intermediate types
  * Better debug output in the C code generator
  * `CBuilder`: Arduino support and improved core functions
  * Normalization functions separated into an independent file
  * `EmbediaFile`: association and injection of definitions into C files
* Improved accuracy in spectrogram implementation
* SVM implementation improvements
  * Improvements in SVM implementation
  * Updated `wrappers/sklearn/svm.py` and `svm_base.py` for abstract types

### Changed

* Documentation
* READMEs in 8 languages: ES, DE, FR, IT, PT, RU, ZH, JA
* Updated English tutorial with a Spanish version
* `CONTRIBUTORS.md` updated
* Memory Management: fully static memory management in project export
* Wrappers: new `sklearn/tree.py` wrapper for `DecisionTreeClassifier`
* Updated `tensorflow_wrappers.py` and `embedia_wrappers.py`

### Fixed

* SVM quant8: typo in preprocessor directive and undefined `dequantize()` function
* `CBuilder`: invalid code when inline layer was in first position
* `type_converters.py`: silent overflows, added counter system
* `swap_alloc_slice`: crash when the first layer changed

## v0.97.0

* Some fixes for sklearn logistic regressor
* Merge pull request #1 from Tomas-De-Blasio-ing/feature/logistic-regression
* A├▒adiendo archivos restantes (falta error de ruas)
* Integracion de LogisticRegression (falta cambio de carpeta para output)
* Integracion de libreria y core de logisticRegression
* Update of common.h headers
* Update scripts after the ProjectFiles option refactoring
* Merge branch 'main' of https://github.com/Embed-ML/EmbedIA
* Update CHANGELOG.md
* Update gitignore
* Core Improvements: Memory Stats, SVM Support, and Code Generation Updates
* Added change-log from git commits
* Update EmbedIA english tutorial
* Add Spanish version
* Added spanish tutorial on EmbedIA
* Some docs update
* major update with full quantization, preprocessing pipeline and sklearn model integration
* Remove mandatory dependency from larq
* EmbedIA class scheme
* Add KNN support for classification based on Scikit-learn for float & fixed32 types
* Refactoring to enable integration of libraries other than TensorFlow
* Refactoring to simplify the calculation of layer/element sizes and parameters.
* Small refactoring to simplify the project generator's operation
* Refactoring to allow associating the different algorithm implementations in different files
* Wrappers for binary Larq layers & EmbedIA spectrogram added
* Significant refactoring to decouple the operation of EmbedIA from the operation of Tensorflow
* Merge of EmbedIA Layer and DataLayer
* TypeConverter small refactoring
* Add ModelFactory class for creating EmbedIA models
* first refactoring of the EmbedIA model was carried out to support other models besides Tensorflow & bugs fixed
* Support for properties of convolutional layer tensors was added to TensorFlow for quant8 data type
* Added support for convolutional layer properties in TensorFlow for fixed8 data type
* Added support for convolutional layer properties in TensorFlow for fixed16 data type

## v0.96.7

* Improved warning message formatting
* Moved quant8 folder to mcu/generic
* Modified CBuilder to enable more natural C code generation
* Added ArduinoBuilder class to generalize Arduino/C print handling
* Reimplemented swap_buffer to use static instead of dynamic memory
* Defined dot_product in common.h/c and refactored dependent code
* Unified prediction output for C/Arduino examples
* Added Doxyfile for documentation
* Added license and contributors file
* Added SVM support for fixed16 and fixed8
* Implement partial memory statistics for input, output, and parameter data
* Add buffer size calculation for the most memory-intensive pipeline layer
* Include memory requirements tracking for input/output data across all layers
* Added change-log from git commits

## v0.95.0

* Update EmbedIA english tutorial
* Add Spanish version
* Added spanish tutorial on EmbedIA
* Some docs update
* major update with full quantization, preprocessing pipeline and sklearn model integration
* Remove mandatory dependency from larq
* EmbedIA class scheme
* Add KNN support for classification based on Scikit-learn for float & fixed32 types
* Refactoring to enable integration of libraries other than TensorFlow
* Refactoring to simplify the calculation of layer/element sizes and parameters.
* Small refactoring to simplify the project generator's operation
* Refactoring to allow associating the different algorithm implementations in different files
* Wrappers for binary Larq layers & EmbedIA spectrogram added
* Significant refactoring to decouple the operation of EmbedIA from the operation of Tensorflow
* Merge of EmbedIA Layer and DataLayer
* TypeConverter small refactoring
* Add ModelFactory class for creating EmbedIA models

## v0.80.0

* first refactoring of the EmbedIA model was carried out to support other models besides Tensorflow & bugs fixed
* Support for properties of convolutional layer tensors was added to TensorFlow for quant8 data type
* se agreg├│ soportes para propiedades de capas convolucionales Tensorflow para tipo de datos fixed8
* se agreg├│ soportes para propiedades de capas convolucionales Tensorflow para tipo de datos fixed16
* Added support for Tensorflow convolutional layer properties for float data type
* added support for DepthwiseConv2D for asymmetric kernel, strides and padding (float data type only)
* added support for SeparableConv2D for asymmetric kernel, strides and padding (float data type only)
* Refactoring of kernel_size of filter_t structure
* Added support for non symmetrical kernels for float data type of conv2d layer
* Added debug information to test functions
* Added support for properties padding & strides for Conv2D tensorflow/keras layer for float type
* Implemented ZeroPadding2D layer & added tests for the BatchNormalization layer
* merge .gitignore
* Path correction of functions test .py files
* some research & example for support fixed point trigonometry functions
* mini_speech_commands download fix
* renaming error fixed

## v0.70.0

* Introduces initial automatic testing for layers, elements, and modules. Fixes numerous bugs discovered during the test run
* Merge branch 'main' of https://github.com/Embed-ML/EmbedIA-dev
* depthwise_conv2d_layer update & bug fixed
* Update TO DO.md
* Create TO DO.md
* Some bugs fixed
* Merge branch 'main' of https://github.com/Embed-ML/EmbedIA
* An EmbedIA version for develop & test

## v0.60.0

* Add spectrogram tool
* Fix warning const float in depthwise_new
* Fix non-square dimension images issue
* Fixed problem with conv2d_layer structure and compilation of depthwise_new biases
* Merge branch 'main' of https://github.com/Embed-ML/EmbedIA
* First implementation of depthwise conv2d layer
* Update project_generator.py
* Merge branch 'main' of github.com:Embed-ML/EmbedIA
* Fix double implementation of LeakyRelu in binary/embedia.c
* Update README.md
* Binary .py files indentation solution
* continuation 1
* Binary .py files indentation solution and Fix declarations in 'for' and 'uint32_t' in argmax function of binary libraries
* Merge pull request #7 from Embed-ML/binaryimprovement
* Merge branch 'main' into binaryimprovement
* New features added
* Update Using_EmbedIA.ipynb
* visualization of examples
* Elimination of i variable declarations in for
* Update model.c
* Translation of comments
* Added a first version of several model formats convertion to TF/Keras
* ReadMe update
* QuantSeparableConv2D layer added. Activation layer fixed.
* Merge pull request #4 from Embed-ML/saavedramarcosdavid-patch-2
* Update Using_EmbedIA.ipynb
* Adding forms in colab
* Merge pull request #3 from Embed-ML/saavedramarcosdavid-patch-1
* Update Using_EmbedIA.ipynb
* Fixed export of std_beta parameter
* Optimized Batch Normalization layer to export two arrays instead of three
* Improved memory allocation when invoking the "predict" function
* Improvement for displaying information in the model generation file
* binary_float16 implementation added
* updated README
* variable name replacement
* Added binary-fixed32 implementation. Fixed bugs in fixed implementations

## v0.50.0

* Improved handling of tensorflow/keras layers not implemented. Updated SeparableConv2D layer.
* Refactoring of layer information. Testing for exportation
* Added full CNN sample, bug fixes, some improvements
* Refactoring of activation functions. Added LeakyReLU activation. Updated old comments
* Readme file update
* Big refactoring. Some new features added
* Update README.md
* Merge branch 'main' of github.com:Embed-Ml/EmbedIA into main
* Update README.md
* EmbedIA in C
* add example_comment

## v0.40.0

* Merge branch 'main' of github.com:Embed-Ml/EmbedIA into main
* Update Using_EmebdIA.ipynb
* Update Using_EmbedIA.ipynb
* Update README.md
* Workflow
* Update README.md

## v0.30.0

* Merge branch 'main' of github.com:Embed-Ml/EmbedIA into main
* Update README.md
* Update create_embedia_project.py
* Colab explaining the use of the framework
* Initial version of the exporter
* includes embedia
* Initial version of the exporter

## Cambios no lanzados (HEAD)

* Update changelog for v0.97.0 release
