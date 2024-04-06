# Lista de Trabajo (en curso o próximo)

 - **Soporte para max_pooling de multiples dimensiones para strides y pool_size**

 - **Implementacion de Capas:**
   - ***Capas Orientadas a Elementos:***
        - Add, Substract, Average, Multiply, Maximum?, Minimum?, Dot
   - ***Capas Orientadas a Elementos:***
     - Add, Average, Dot, Substract, Maximum?, Minimum?, Multiply, 
   - ***Capas de Transformación:***
     - Agregar soporte para Capas de especiales que existen solo en entrenamiento
     - Reescaling, Reshape, Resizing, CenterCrop, Concatenate, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
 - **Bugs:**
   - revisar información de número de parametros al mostrar informacion del modelo que aparece en 0
   - revisar cálculo de cantidad de memoria requerida por capas
 - **Modelos EmbedIA:**
   - procesar modelos de modelos en tensorflow 
 - **reestructurar carpetas/archivos Embedia**
   - analizar archivo embedia.h/.c para determinar que queda en neural_nets.h/.c y si hay que crear common.h/.c (y que poner) 
 - **Cuantizacion:**
   - implementar  cuantización por clustering
   - implementar versiones de computo para fixed32 y fixed16

# Lista de tareas (sin orden particular)

## Modelos de Aprendizaje Automático
  - **Hacer ejemplo de Refactoring de EmbedIA para soportar modelos de SkLearn**
    - reestructurar carpetas pensando en los tipos de algoritmos por ejemplo embedia/nn/ o embedia/ml o embedia/svm
    
- **Incorporar modelos de aprendizaje automático basado en ML (Sklearn)**
    - agregar modelos SKLEARN

## Modelos de Redes Neuronales
  - **Incorporar modelos (livianos) populares**
  
  - **agregar capas TF/Keras**
    - Capas Convolucionales/Deconvoluciones:
      - SeparableConv1D, Conv1D, Conv3D, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, DepthwiseConv1D
      
    - Capas Orientadas a Elementos:
      - Add, Average, Dot, Substract, Maximum?, Minimum?, Multiply, 
    - Capas de Transformación:
      - Agregar soporte para Capas de especiales que existen solo en entrenamiento:
      - Capas Estandar:
        - Reescaling, Reshape, Resizing, CenterCrop, Concatenate, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D 
        
    - Capas de Pooling:
      - revisar soporte de capas para Average, max y min pooling con 1D, 2D y 3D
      - Agregar Capas MaxGlobal y AverageGlobal para !D, 2D y 3D
    - Capas de Activación:
      - softplus(x) = log(exp(x) + 1)
      - swish(x) = x * sigmoid(x).
      - selu(x) = 
         if x > 0: return scale * x
         if x < 0: return scale * alpha * (exp(x) - 1)
      - mish(x) = x * tanh(softplus(x))
      - hardsigmoid(x) =
         if x < -2.5: return 0
         if x >  2.5: return 1
         if -2.5 <= x <= 2.5: return 0.2 * x + 0.5 
      - gelu(x, approximate) = 
         0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) if approximate is True 
         x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2))), where P(X) ~ N(0, 1), if approximate is False. 
      - exponential(x) = exp(x). 
      - elu(x) = 
         x if x > 0
         alpha * (exp(x) - 1) if x < 0. 
    - Capas Especiales:
      - GroupNormalization
      - Lambda: aplica funcion en python a elementos
      
    

  - **agregar soporte para channel_last/first**
    - agregar versiones de versiones con la variante channel_first y channel_last (embedia trabaja con channel first y adapta la entrada para trabajar de esa manera). Considerar usar el mismo orden que tensorflow por simplicidad (mucho refactoring)
  
  - **Pensar alternativas y posibilidades de modelos no secuenciales**
    
  - **Agregar soporte para incorporar modelos completos dentro de modelos de TF/Keras**
    - En TF/Keras, un modelo puede tener otro modelo con capas. Es comun en autoencoders o en modelos grandes donde se repiten bloques.

## Soporte para funciones de señales, audio y video
  - **incorporar funciones para imagenes**
    - agregar algoritmo de componentes conectados, conversión color y escala de imágenes. 

    
## Otros/Varios
  - **Actualiza documentación**
    - agregar propiedad project_files en documentacion
    - agregar el paper de smart-tech como referencia como manera de "citar embedia"
    - licencia de software? MIT?
       
  - **Hacer un paquete auto/instalable**
    - Armar un paquete de instalación
    - Agregar paquete a repositorio.
    
  - **Mejorar y organizar los ejemplos**
    - Brainstorming sobre esto. Se podrian craer carpetas con ejemplos y dentro de cada carpeta diferentes soluciones (CNN, NN, Autoencoders, SVM, etc.).

  - **Generar modelos robustos para tareas específicas**
    - Analizar problemas, conseguir datos y generar modelos para señales, imagenes y sonido.
      
  - **Incorporar Microcontroladores**
    - Ver que se puede hacer. Incorporar RP2040 Wifi y ARM M4F.

  - **Optimización para Microcontroladores específicos**
    - Analizar modelos exportados de EdgeImpulse
    - Ver proyecto chino (no recuerdo cual) que incorporaba este muchos tipos de MCU y optimizaciones
     
  - **Refactoring varios**
    - en C para orientar a operaciones tipo multiplicacion de matrices.
    - en python para crear un modelo embedia que se pueda transformar a c. Luego cualquier otro formato se pasa a EmbedIA. Esto facilitaria la incorporación de otros formatos de modelos.    

  - **Analizar mejora para depuración**
    - Para debug incluir opciones con potencia binaria Ej: 0 = no imprimir, 1= encabezado, 2 valores de salida, 4=pesos de capa?

  - **Unificar tipos de datos**
    - Tal vez un datatypes.h con todas las definiciones (fixed8, fixed16, fixed8, quant8, binary, etc.)
    - En embedia.h definir estructuras de datos para todos los tipos(aunque no se usen) vector, matriz, tensor, etc. estos tal vez podrían ir en datatypes.h y dejar solo los tipos especificos en embedia ( que luego tendrá su propia version, embedia_nn.h, embedia_svm.h, lo que sea).
    - unificar en carpetas archivos de implementación similar de datos y renombrar los archivos. Ej en "fixed" tendriamos fixed32.h, fixed16.h y fixed8.h
   
  - **Incorporar guardas C++ (extern) a los archivos de C**
   
  
## BUGS
- **Cantidad de bits para tipos de datos fixed**
  - la clase model crea los conversores de float a otro tipo de datos. En particular para los tipos de datos fixed, podria haber problemas si no es consistente con los valores declarados en el archivo fixed.h. 
    - Posible solución: asignar estos valores en los archivos fixed.h al general el proyecto embedia 

- Revisar la funcion Tanh para fixed16 y fixed32. Las pruebas indican un error mayor al de validación

- Revisar DepthwiseConv2D para kernel 2x3 y tipo quant8

- Agregar test para Flatten
- Agregar test para Binarias
- Agregar Test para Espectrograma

- 
## Optimizaciones:
- **Velocidad**
  - para incrementar la velocidad de procesamiento podrían realizarse optimizaciones de código específicas teniendo en cuenta los parámetros. Algo así se implemento con la capa Conv2D que tiene versiones sin padding ni strides, sin padding con strides y con padding y strides.
  - Analizar la posibilidad de introducir para procesadores especiales optimizaciones basadas en funciones que realizan operaciones con punteros sobre datos.
  - 

## LISTOS:
 - **Soporte de strides para capas convolucioneales: SeparableConv2D y DepthwiseConv2D**
 - **Soporte de padding para capas convolucioneales: Conv2D(ya implementada para float), SeparableConv2D y DepthwiseConv2D**
 - **Soporte para kernels asimetricos para Conv2D**
 - **Refactoring de propiedad kernel_size para llevarla a conv2d_layer desde filter_T**
 - **Refactoring para dividir Model en EmbediaModel y TensorflowModel**
 - **Refactoring para posibilitar modelos basados en Scikit Learn**
 - **Crear una arquitectura de clases para incorporar Modelos ML de Sklearn a EmbedIA.**
 - **Modificar EmbedIA para incluir los modelos ML de Sklearn.**
 - **agregar padding y stride en capas convolucionales**
 - **Refactoring para que cada "capa"/elemento/modelo de EmbedIA indique los archivos C/C++ que requiere**
  - las capas especializadas como de procesamiento general podrían requerir de archivos especiales que deberían ser incluídos. Esto debería delegarse a la capa.
 - ** eliminar calculo de tamaño de capas/estructuras. Actualmente se analiza código C y es muy sensible a errores
 - **Refactoring de EmbedIA para soportar modelos de SkLearn** (p)
 - (Se mantuvo en gral nombres actuales) refactoring de propiedades asociadas a TF/Keras para generalizacion. Ej: en general usamos layer como elemento que contiene parte del modelo, podría utilizarse component, element, block, module, etc.

      
