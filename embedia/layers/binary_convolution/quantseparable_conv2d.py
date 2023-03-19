from embedia.layers.data_layer import DataLayer
from embedia.model_generator.project_options import ModelDataType
from embedia.utils import file_management
from embedia.utils.binary_helper import BinaryGlobalMask
from embedia.model_generator.project_options import BinaryBlockSize
import numpy as np
import larq as lq
import math

class QuantSeparableConv2D(DataLayer):

    def __init__(self, model, layer, options, **kwargs):

        super().__init__(model, layer, options, **kwargs)
        # the type defined in "struct_data_type" must exists in "embedia.h"
        # self.struct_data_type = self.get_type_name().lower()+'_layer_t'
        self.input_data_type = "data3d_t"
        self.output_data_type = "data3d_t"
        self.depth_weights = self.adapt_weights(layer.get_weights()[0])
        self.point_weights = self.adapt_weights(layer.get_weights()[1])
        self.biases = layer.get_weights()[2]

        #verificamos a que caso corresponde
        with lq.context.quantized_scope(True):
            if (layer.get_config()['input_quantizer'] == None) and (layer.get_config()['pointwise_quantizer'] == None) and (layer.get_config()['depthwise_quantizer'] == None):
                
                #es una conv normal
                self.tipo_conv = 0
                    
            elif (layer.get_config()['input_quantizer'] != None) and (layer.get_config()['pointwise_quantizer'] != None) and (layer.get_config()['depthwise_quantizer'] != None):
                if (layer.get_config()['input_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['pointwise_quantizer']['class_name'] == 'SteSign') and (layer.get_config()['depthwise_quantizer']['class_name'] == 'SteSign'):
                    #conv pura binaria 
                    self.tipo_conv = 1
                else:
                    print(f"Error: No support for layer with this arguments")
                    raise f"Error: No support for layer with this arguments"
            else:
                print(f"Error: No support for layer with this arguments")
                raise f"Error: No support for layer with this arguments"


    def var(self):
        if self.tipo_conv == 0:
            
            return f"separable_conv2d_layer_t {self.name}_data;\n"
        
        else:
            
            return f"quant_separable_conv2d_layer_t {self.name}_data;\n"


    def prototypes_init(self):
        if self.tipo_conv == 0:
            
            return f"separable_conv2d_layer_t init_{self.name}_data(void);\n"
        else:
            return f"quant_separable_conv2d_layer_t init_{self.name}_data(void);\n"


    def adapt_weights(self, weights):
        _row, _col, _can, _filt = weights.shape
        arr = np.zeros((_filt, _can, _row, _col))
        for row, elem in enumerate(weights):
            for col, elem2 in enumerate(elem):
                for chn, elem3 in enumerate(elem2):
                    for filt, value in enumerate(elem3):
                        arr[filt, chn, row, col] = value
        return arr

 
    def calculate_MAC(self):

        out_size = self.get_output_size()

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self.depth_weights.shape
        MACs = out_size*n_cols*n_rows*n_channels

        n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        MACs += out_size*n_cols*n_rows*n_channels

        return MACs

    def calculate_memory(self, types_dict):
        """
        calculates amount of memory required to store the data of layer
        Returns
        -------
        int
            amount memory required

        """

        # layer dimensions
        n_channels, n_filters, n_rows, n_cols = self.depth_weights.shape
        depth_params = n_channels * n_filters * n_rows * n_cols

        n_channels, n_filters, n_rows, n_cols = self.point_weights.shape
        point_params = n_channels * n_filters * n_rows * n_cols

        # base data type in bits: float, fixed (32/16/8)
        dt_size = ModelDataType.get_size(self.options.data_type)

        if (self.tipo_conv==0):

            # EmbedIA filter structure size
            sz_filter_t = types_dict['filter_t']

            mem_size = ((depth_params * dt_size / 8 + sz_filter_t) + ((point_params * dt_size / 8 + sz_filter_t) * n_filters))

        else:
            # EmbedIA filter structure size
            sz_filter_t = types_dict['quant_filter_t']

            if self.options.tamano_bloque == BinaryBlockSize.Bits8:
                dt_size = 8
            elif self.options.tamano_bloque == BinaryBlockSize.Bits16:
                dt_size = 16
            elif self.options.tamano_bloque == BinaryBlockSize.Bits32:
                dt_size = 32
            else:
                dt_size = 64

            mem_size = ((math.ceil(depth_params/dt_size) * dt_size / 8 + sz_filter_t) + ((math.ceil(point_params/dt_size) * dt_size / 8 + sz_filter_t) * n_filters))

        return mem_size


    def functions_init(self):
        depth_filtros, depth_channels, depth_rows, depth_columns = self.depth_weights.shape  # Getting layer info from it's weights
        assert depth_rows == depth_columns  # WORKING WITH SQUARE KERNELS FOR NOW
        depth_kernel_size = depth_rows  # Defining kernel size

        point_filters, point_channels, _, _ = self.point_weights.shape  # Getting layer info from it's weights

        struct_type = self.struct_data_type
        (data_type, macro_converter) = self.model.get_type_converter()

        if self.tipo_conv==0:   #separable normal

            init_conv_layer = f'''

            {struct_type} init_{self.name}_data(void){{

            '''
            o_weights = ""
            for ch in range(depth_channels):
                for f in range(depth_rows):
                    o_weights += '\n    '
                    for c in range(depth_columns):
                        o_weights += f'''{macro_converter(self.depth_weights[0,ch,f,c])}, '''

                o_weights += '\n  '

            o_code = f'''
            static const {data_type} depth_weights[]={{{o_weights}
            }};
            static filter_t depth_filter = {{{depth_channels}, {depth_kernel_size}, depth_weights, 0}};

            static filter_t point_filters[{point_filters}];
            '''
            init_conv_layer += o_code
            
            for i in range(point_filters):
                o_weights = ""
                for ch in range(point_channels):
                    o_weights+=f'''{macro_converter(self.point_weights[i,ch,0,0])}, '''

                o_code = f'''
            static const {data_type} point_weights{i}[]={{{o_weights}
            }};
            static filter_t point_filter{i} = {{{point_channels}, 1, point_weights{i}, {macro_converter(self.biases[i])}}};
            point_filters[{i}] = point_filter{i};
            '''
                init_conv_layer += o_code

            init_conv_layer += f'''
            {struct_type} layer = {{{point_filters}, depth_filter, point_filters}};
            return layer;
            }}
            '''
        
        else:   #separable binaria

            struct_type = 'quant_separable_conv2d_layer_t'

            largo_total = (depth_rows)*(depth_columns)

            largo_total_point = point_channels

            if self.options.tamano_bloque == BinaryBlockSize.Bits8:
                xBits = 8
                block_type = 'uint8_t'
            elif self.options.tamano_bloque == BinaryBlockSize.Bits16:
                xBits = 16
                block_type = 'uint16_t'
            elif self.options.tamano_bloque == BinaryBlockSize.Bits32:
                xBits = 32
                block_type = 'uint32_t'
            else:
                xBits = 64
                block_type = 'uint64_t'

            init_conv_layer = f'''

            {struct_type} init_{self.name}_data(void){{

            '''
            o_weights = ""
            for ch in range(depth_channels):
                cont = 0
                suma = 0
                for f in range(depth_rows):
                    o_weights += '\n    '
                    for c in range(depth_columns):
                        num = self.depth_weights[0,ch,f,c]
                        if xBits==16:
                          if num == 1.0:  
                              suma += (BinaryGlobalMask.get_mask_16())[cont]
                        elif xBits==32:
                          if num == 1.0: 
                              suma += (BinaryGlobalMask.get_mask_32())[cont]
                        elif xBits==64:
                          if num == 1.0: 
                              suma += (BinaryGlobalMask.get_mask_64())[cont]
                        else:
                          if num == 1.0: 
                              suma += (BinaryGlobalMask.get_mask_8())[cont]

                        if cont == xBits-1 or ((f+1)*(c+1) == largo_total):
                          o_weights+=f'''{(suma)},'''
                          cont = 0
                          suma = 0
                        else:
                          cont+=1

                o_weights += '\n  '

            o_code = f'''
            static const {block_type} depth_weights[]={{{o_weights}
            }};
            static quant_filter_t depth_filter_b = {{{depth_channels}, {depth_kernel_size}, depth_weights, 0}};

            static quant_filter_t point_filters_b[{point_filters}];
            '''
            init_conv_layer += o_code
            
            for i in range(point_filters):
                cont = 0
                suma = 0
                o_weights = ""
                for ch in range(point_channels):
                    num = self.point_weights[i,ch,0,0]
                    if xBits==16:
                        if num == 1.0:  
                            suma += (BinaryGlobalMask.get_mask_16())[cont]
                    elif xBits==32:
                        if num == 1.0: 
                            suma += (BinaryGlobalMask.get_mask_32())[cont]
                    elif xBits==64:
                        if num == 1.0: 
                            suma += (BinaryGlobalMask.get_mask_64())[cont]
                    else:
                        if num == 1.0: 
                            suma += (BinaryGlobalMask.get_mask_8())[cont]

                    if cont == xBits-1 or ((ch+1) == largo_total_point):

                        o_weights+=f'''{(suma)},'''
                        cont = 0
                        suma = 0
                    else:
                        cont+=1

                o_code = f'''
            static const {block_type} point_weights{i}[]={{{o_weights}
            }};
            static quant_filter_t point_filter{i} = {{{point_channels}, 1, point_weights{i}, {macro_converter(self.biases[i])}}};
            point_filters_b[{i}] = point_filter{i};
            '''
                init_conv_layer += o_code

            init_conv_layer += f'''
            {struct_type} layer = {{{point_filters}, depth_filter_b, point_filters_b}};
            return layer;
            }}
            '''

        return init_conv_layer
    
    def predict(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "embedia.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
        "embedia.c"

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "embedia.c".

        """
        if self.tipo_conv==0:
            return f'''    separable_conv2d_layer({self.name}_data, {input_name}, &{output_name});
'''
        else:
            return f'''    quantSeparableConv2D_layer({self.name}_data, {input_name}, &{output_name});
'''
