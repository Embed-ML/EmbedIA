import numpy as np
from tensorflow import keras

from embedia.project_options import ModelDataType, ProjectType, ProjectFiles, DebugMode

def convertir_pesos(weights):
  '''
    Used internally to transpose weights 4 dimentions array. Our library works with weights with the form (filter,channel,row,column)
    It goes from (row,column,channel,filter) to (filter,channel,row,column)
    For example: It goes from (3,3,1,8) to (8,1,3,3)
    Receives: weights from keras/tf model (model.get_weights return)
    Returns: weights our library can work with
  '''
  _fila,_col,_can,_filt = weights.shape
  arr = np.zeros((_filt,_can,_fila,_col))
  for fila,elem in enumerate(weights):
    for columna,elem2 in enumerate(elem):
      for canal,elem3 in enumerate(elem2):
        for filtros,valor in enumerate(elem3):
          #print("F:{0}, C:{1}, Canal:{2}, Filtro:{3} -> Valor: {4}".format(fila,columna,canal,filtro,valor))
          arr[filtros,canal,fila,columna] = valor   
  return arr

def get_weights_separable(layer):
  '''
    Function to return model weights from cnn layer in a way our model can work with
    Params:
    layer -> convolutional layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (filters,channels,rows,columns)
    biases -> array with dimention: (filters)

    example of usage: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'separable_conv2d' in layer.name #asserting I didn't receive a non convolutional layer
  depth_weights=layer.get_weights()[0]
  point_weights=layer.get_weights()[1]
  biases=layer.get_weights()[2]
  
  depth_weights=convertir_pesos(depth_weights)
  point_weights=convertir_pesos(point_weights)
  
  return depth_weights,point_weights,biases


def get_weights_cnn(layer):
  '''
    Function to return model weights from cnn layer in a way our model can work with
    Params:
    layer -> convolutional layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (filters,channels,rows,columns)
    biases -> array with dimention: (filters)

    example of usage: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'conv2d' in layer.name #asserting I didn't receive a non convolutional layer
  weights=layer.get_weights()[0]
  biases=layer.get_weights()[1]
  weights=convertir_pesos(weights)
  return weights,biases


def exportar_separable_a_c(layer,nro, macro_converter, data_type):
  '''
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model separable weights
  '''
  
  depth_weights,point_weights,biases=get_weights_separable(layer)


  depth_filtros,depth_canales,depth_filas,depth_columnas=depth_weights.shape #Getting layer info from it's weights
  assert depth_filas==depth_columnas #WORKING WITH SQUARE KERNELS FOR NOW
  depth_kernel_size=depth_filas #Defining kernel size

  point_filtros,point_canales,_,_=point_weights.shape #Getting layer info from it's weights

  ret=""

  init_conv_layer=f'''

separable_layer_t init_separable_layer{nro}(void){{

  '''
  o_weights=""
  for ch in range(depth_canales):
    for f in range(depth_filas):
      o_weights+='\n    '
      for c in range(depth_columnas):
        o_weights+=f'''{macro_converter(depth_weights[0,ch,f,c])}, '''
      #o_weights+='\n'
    o_weights+='\n  '

  o_code=f'''
  static const {data_type} depth_weights[]={{{o_weights}
  }};
  static filter_t depth_filter = {{{depth_canales}, {depth_kernel_size}, depth_weights, 0}};

  static filter_t point_filters[{point_filtros}];
  '''
  init_conv_layer+=o_code
  
  for i in range(point_filtros):
    o_weights=""
    for ch in range(point_canales):
      o_weights+=f'''{macro_converter(point_weights[i,ch,0,0])}, '''
    
    o_code=f'''
  static const {data_type} point_weights{i}[]={{{o_weights}
  }};
  static filter_t point_filter{i} = {{{point_canales}, 1, point_weights{i}, {macro_converter(biases[i])}}};
  point_filters[{i}]=point_filter{i};
  '''
    init_conv_layer+=o_code
  
  init_conv_layer+=f'''
  separable_layer_t layer = {{{point_filtros},depth_filter,point_filters}};
  return layer;
}}
  '''

  ret+=init_conv_layer

  return ret


def exportar_cnn_a_c(layer,nro, macro_converter, data_type):
  '''
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model cnn weights
  '''
  
  pesos,biases=get_weights_cnn(layer)


  filtros,canales,filas,columnas=pesos.shape #Getting layer info from it's weights

  assert filas==columnas #WORKING WITH SQUARE KERNELS FOR NOW
  kernel_size=filas #Defining kernel size

  ret=""

  init_conv_layer=f'''

conv_layer_t init_conv_layer{nro}(void){{

  static filter_t filtros[{filtros}];
  '''
  for i in range(filtros):
    o_weights=""
    for ch in range(canales):
      for f in range(filas):
        o_weights+='\n    '
        for c in range(columnas):
          o_weights+=f'''{macro_converter(pesos[i,ch,f,c])}, '''
        #o_weights+='\n'
      o_weights+='\n  '
 
    o_code=f'''
  static const {data_type} weights{i}[]={{{o_weights}
  }};
  static filter_t filter{i} = {{{canales}, {kernel_size}, weights{i}, {macro_converter(biases[i])}}};
  filtros[{i}]=filter{i};
    '''
    init_conv_layer+=o_code
  init_conv_layer+=f'''
  conv_layer_t layer = {{{filtros},filtros}};
  return layer;
}}
  '''

  ret+=init_conv_layer

  return ret


def get_weights_dense(layer):
  '''
    Function to return model weights from dense layer in a way our model can work with
    Params:
    layer -> dense layer
    
    Returns:
    Tuple with values: weights y biases
    weights -> array with dimentions: (input,neurons)
    biases -> array with dimention: (filters)

    Example: weights,bias=get_weights_cnn(layerCNN)
  '''
  assert 'dense' in layer.name #Get sure it is a dense layer
  weights=layer.get_weights()[0]
  biases=layer.get_weights()[1]
  return weights,biases

def exportar_densa_a_c(layer,nro, macro_converter, data_type):
  '''
    Builds embedia's init_dense_layer function
    Receives
    layer          --> instance of a layer from a model (model.layers[i])
    nro            --> from input to output, the number corresponding to the position of this layer
    macro_converter--> a macro used if working with embedia fixed. Adds macro to numbers in c code
    data_type      --> 'float' or 'fixed' depending embedia optinons

    Returns:
    String with c code representing the function with model dense weights
  '''
  pesos,biases=get_weights_dense(layer)
  input,neuronas=pesos.shape
  ret=""

  init_dense_layer=f'''
dense_layer_t init_dense_layer{nro}(){{
  // Cantidad de variables weights = numero de neuronas
  // Cantidad de pesos por weights = numero de entradas

  static neuron_t neuronas[{neuronas}];
  '''
  o_code=""
  for neurona in range(neuronas):
    o_weights=""
    #for p in pesos[neurona,:]:
    for p in pesos[:,neurona]:
      o_weights+=f'''{macro_converter(p)}, '''
    o_weights=o_weights[:-1] #remuevo la ultima coma
    #o_weights+='\n'
    o_code+=f'''
  static const {data_type} weights{neurona}[]={{
    {o_weights}
  }};
  static neuron_t neuron{neurona} = {{weights{neurona}, {macro_converter(biases[neurona])}}};
  neuronas[{neurona}]=neuron{neurona};
    '''
  init_dense_layer+=o_code

  init_dense_layer+=f'''
  dense_layer_t layer= {{{neuronas}, neuronas}};
  return layer;
}}
'''
  return init_dense_layer


#=================================

#CREATE MODEL INIT FUNCTION
def create_model_init(cantConv,cantDensas,cantSeparable):
  #Begin model_init function string
  model_init = f'''
void model_init(){{
'''
  for i in range(cantSeparable):
    model_init+=f'''    separable_layer{i} = init_separable_layer{i}(); // Capa depthwise separable conv {i+1}\n'''

  for i in range(cantConv):
    model_init+=f'''    conv_layer{i} = init_conv_layer{i}(); // Capa convolucional {i+1}\n'''

  for i in range(cantDensas):
    model_init+=f'''    dense_layer{i} = init_dense_layer{i}(); //Capa densa {i+1}\n'''
  
  model_init+=f'''}}\n'''
  #End of model_init function string
  return model_init

#==============================

#MODEL PREDICT FUNCTION UTILS

def model_predict_separable(layer,index,isFirst,layerNumber,options):

  #me aseguro que tenga una activacion implementada
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

  ret=f'''
  // Capa {layerNumber}: Depthwise Separable Conv2D
  separable_conv2d_layer(separable_layer{index},input,&output);
    '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Output matrix layer {layerNumber} (Depthwise Separable Conv2D): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  if layer.activation==keras.activations.relu:
    ret+=f'''// Activation Layer {layerNumber}: relu
  relu(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Relu): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''// Activation Layer {layerNumber}: tanh
  tanh2d(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  ret+='''input=output;
  '''
  return ret



def model_predict_conv(layer,index,isFirst,layerNumber,options):

  #me aseguro que tenga una activacion implementada
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.tanh

  ret=f'''
  // Capa {layerNumber}: Conv 2D
  conv2d_layer(conv_layer{index},input,&output);
    '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Output matrix layer {layerNumber} (Conv 2D): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  if layer.activation==keras.activations.relu:
    ret+=f'''// Activation Layer {layerNumber}: relu
  relu(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Relu): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''// Activation Layer {layerNumber}: tanh
  tanh2d(output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_data_t("Activation Layer {layerNumber} (Tanh): ", output);
      #endif // EMBEDIA_DEBUG
      '''
  ret+='''input=output;
  '''
  return ret



def model_predict_maxPool(pool_size, stride,layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: MaxPooling2D
  max_pooling_2d({pool_size},{stride},input,&output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
    #if EMBEDIA_DEBUG > 0
    print_data_t("Output Layer  {layerNumber} (MaxPooling 2D): ", output);
    #endif // EMBEDIA_DEBUG
    '''
  ret+='''input=output;
  '''
  return ret

def model_predict_avgPool(pool_size, stride,layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: AveragePooling2D
  avg_pooling_2d({pool_size},{stride},input,&output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
    #if EMBEDIA_DEBUG > 0
    print_data_t("Output Layer  {layerNumber} (AveragePooling 2D): ", output);
    #endif // EMBEDIA_DEBUG
    '''
  ret+='''input=output;
  '''
  return ret

def model_predict_flatten(layerNumber,options):
  ret = f'''
  // Capa {layerNumber}: Flatten
  flatten_data_t f_output;
  flatten_layer(output, &f_output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Output Vector Layer {layerNumber} (Flatten): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  ret += '''f_input=f_output;
  '''
  return ret

def model_predict_dense(layer,index,layerNumber,options,isLastLayer=False):
  assert layer.activation==keras.activations.relu or layer.activation==keras.activations.softmax or layer.activation==keras.activations.tanh
  ret = f'''
  // Capa {layerNumber}: Dense
  dense_forward(dense_layer{index},f_input,&f_output);
  '''
  if options.debug_mode != DebugMode.DISCARD:
    ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Output Vector Layer {layerNumber} (Dense): ",f_output);
      #endif // EMBEDIA_DEBUG
    '''
  if layer.activation==keras.activations.relu:
    ret+=f'''
  //Activación Capa {layerNumber}: relu
  relu_flatten(f_output);
  '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Relu): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.softmax:
    ret+=f'''
  //Activación Capa {layerNumber}: softmax
  softmax(f_output);
    '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Softmax): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  elif layer.activation==keras.activations.tanh:
    ret+=f'''
  //Activación Capa {layerNumber}: tanh
  tanh_flatten(f_output);
  '''
    if options.debug_mode != DebugMode.DISCARD:
      ret+=f'''
      #if EMBEDIA_DEBUG > 0
      print_flatten_data_t("Activation Layer {layerNumber} (Tanh): ", f_output);
      #endif // EMBEDIA_DEBUG
      '''
  if (not isLastLayer):
    ret+='''f_input = f_output;
    '''
  return ret



#CREATE MODEL PREDICT FUNCTION
def create_model_predict(model,options):
  ret='''
int model_predict(data_t input, flatten_data_t * results){
  data_t output;
  flatten_data_t f_input;
  '''
  if options.debug_mode != DebugMode.DISCARD:
      ret+='''
      // Input
          #if EMBEDIA_DEBUG > 0
          print_data_t("Input data:", input);
          #endif // EMBEDIA_DEBUG
      '''

  cantSeparable=0
  cantConv=0
  cantDensas=0

  for i,layer in enumerate(model.layers):
    if 'separable_conv2d' in layer.name:
      ret+=model_predict_separable(layer,cantSeparable,i==0,i+1,options)
      cantSeparable+=1
    elif 'conv2d' in layer.name:
      ret+=model_predict_conv(layer,cantConv,i==0,i+1,options)
      cantConv+=1
    elif 'dense' in layer.name:
      ret+=model_predict_dense(layer,cantDensas,i+1,options,i+1==len(model.layers))
      cantDensas+=1
    elif 'max_pooling2d' in layer.name:
      pool_size,_=layer.pool_size
      stride,_=layer.strides
      ret+=model_predict_maxPool(pool_size,stride,i+1,options)
    elif 'average_pooling2d' in layer.name:
      pool_size,_=layer.pool_size
      stride,_=layer.strides
      ret+=model_predict_avgPool(pool_size,stride,i+1,options)
    elif 'flatten' in layer.name:
      ret+=model_predict_flatten(i+1,options)
    else:
      return f"Error: No support for layer {layer} which is of type {layer.activation}"

  ret+='''
  int result= argmax(f_output);
  *results = f_output;
  return result;
}
  '''
  return ret



def create_codeblock_project(project_name,files):
    output=f'''
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<CodeBlocks_project_file>
	<FileVersion major="1" minor="6" />
	<Project>
		<Option title="{project_name}" />
		<Option pch_mode="2" />
		<Option compiler="gcc" />
		<Build>
			<Target title="Debug">
				<Option output="bin/Debug/{project_name}" prefix_auto="1" extension_auto="1" />
				<Option object_output="obj/Debug/" />
				<Option type="1" />
				<Option compiler="gcc" />
				<Compiler>
					<Add option="-g" />
				</Compiler>
			</Target>
			<Target title="Release">
				<Option output="bin/Release/{project_name}" prefix_auto="1" extension_auto="1" />
				<Option object_output="obj/Release/" />
				<Option type="1" />
				<Option compiler="gcc" />
				<Compiler>
					<Add option="-O2" />
				</Compiler>
				<Linker>
					<Add option="-s" />
				</Linker>
			</Target>
		</Build>
		<Compiler>
			<Add option="-Wall" />
		</Compiler>'''
    for filename in files:
        if filename[-2:].lower()=='.c':
            output+=f'''
		<Unit filename="{filename}">
			<Option compilerVar="CC" />
		</Unit>'''
        elif filename[-2:].lower()=='.h':
            output+=f'''
		<Unit filename="{filename}" />'''

    output+='''
		<Extensions>
			<code_completion />
			<envvars />
			<debugger />
		</Extensions>
	</Project>
</CodeBlocks_project_file>'''
    return output
