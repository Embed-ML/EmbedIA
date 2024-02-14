# Lista de tareas (sin orden particular)

## Modelos de Aprendizaje Automático
  - **Refactoring de EmbedIA para soportar modelos de SkLearn**
    - reestructurar carpetas pensando en los tipos de algoritmos por ejemplo embedia/nn/ o embedia/ml o embedia/svm
    - refactoring de propiedades asociadas a TF/Keras para generalizacion. Ej: en general usamos layer como elemento que contiene parte del modelo, podría utilizarse componente, elemento, etc.
    
- **Incorporar modelos de aprendizaje automático basado en ML (Sklearn)**
  - Crear una arquitectura de clases para incorporar Modelos ML de Sklearn a EmbedIA.
  - Modificar EmbedIA para incluir los modelos ML de Sklearn.

## Modelos de Redes Neuronales
  - **Incorporar modelos (livianos) populares**
  
  - **agregar capas TF/Keras**
    - Convoluciones, Deconvoluciones, UpSample, Capas de Transformación
    
  - **agregar padding en capas convolucionales**
  
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
     
  - **Refactoring para que cada "capa"/elemento de EmbedIA indique los archivos C/C++ que requiere**
    - las capas especializadas como de procesamiento general podrían requerir de archivos especiales que deberían ser incluídos. Esto debería delegarse a la capa.
        
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
   
  


            
      
