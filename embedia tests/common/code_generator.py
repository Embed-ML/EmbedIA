from enum import Enum
import os
import glob
import shutil
import numpy as np

from embedia.models import ModelFactory
from embedia.layers.activation.activation import Activation
from embedia.layers.data_layer import DataLayer
from embedia.model_generator.project_options import ProjectOptions, ModelDataType
from embedia.utils.file_management import save_to_file

class TestResult(Enum):
    ERROR = "ERROR"
    FAIL = "FAIL"
    SUCCESS = "SUCCESS"


class CodeGenerator:
    def __init__(self, embedia_path, output_path):
        self._embedia_path = embedia_path
        self._output_path = output_path
        self._embedia_type = None
        self._project_name = 'unkown'

    def set_embedia_type(self, embedia_type):
        self._embedia_type = embedia_type

    def set_project_name(self, project_name):
        self._project_name = project_name

    def get_project_folder(self):
        return os.path.abspath(os.path.join(self._output_path, self._project_name))

    def _copy_file_with_replace(self, src_file, dst_file, replace_dict):
        with open(src_file, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                for key, value in replace_dict.items():
                    new_ln = line.replace(key, value)
                    if new_ln != line:
                        lines[i] = new_ln
                        break

        with open(dst_file, 'w') as file:
            file.writelines(lines)


    def _copy_embedia_files(self):
        source_path = os.path.abspath(os.path.join(self._embedia_path, 'libraries'))
        dest_path = os.path.abspath(self._output_path)

        for datatype in [ModelDataType.FLOAT, ModelDataType.FIXED32, ModelDataType.FIXED16, ModelDataType.FIXED8,
                         ModelDataType.QUANT8]:
            folder = ModelDataType.get_name(datatype)
            src_folder = os.path.join(source_path, folder)
            dst_folder = os.path.join(dest_path, 'embedia', folder)
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

        src_folder = os.path.join(source_path, 'debug')
        dst_folder = os.path.join(dest_path, 'embedia', 'debug')
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

        filename = os.path.join(dest_path, 'embedia', 'debug', 'embedia_debug.h')

        if self._embedia_type == ModelDataType.QUANT8:
            datatype_folder = ModelDataType.get_name(ModelDataType.FLOAT)
        else:
            datatype_folder = ModelDataType.get_name(self._embedia_type)
        repl_dict = {
            '{include}': '#include "embedia_debug.h"',
            '{EMBEDIA_DEBUG}': '#define EMBEDIA_DEBUG 2',
            'embedia.h': f'../{datatype_folder}/embedia.h'
        }
        self._copy_file_with_replace(filename, filename, repl_dict)

        filename = os.path.join(dest_path, 'embedia', 'debug', 'embedia_debug.c')
        repl_dict = {
            'embedia_debug_def.h': 'embedia_debug_def_c.h'
        }
        self._copy_file_with_replace(filename, filename, repl_dict)




    def _create_project_folder(self):
        output_folder =  self.get_project_folder()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def _create_file(self, rel_filename, content):

        save_to_file(os.path.join(self.get_project_folder(), 'main.c'), content)

    def _get_embedia_include_headers(self):
        if self._embedia_type == ModelDataType.FIXED32:
            type_file = 'fixed32/fixed'
            embedia_file = 'fixed32/embedia'
        elif self._embedia_type == ModelDataType.FIXED16:
            type_file = 'fixed16/fixed'
            embedia_file = 'fixed16/embedia'
        elif self._embedia_type == ModelDataType.FIXED8:
            type_file = 'fixed8/fixed'
            embedia_file = 'fixed8/embedia'
        elif self._embedia_type == ModelDataType.QUANT8:
            type_file = 'quant8/quant8'
            embedia_file = 'quant8/embedia'
        else:
            type_file = ''
            embedia_file = 'float/embedia'

        headers = ['../embedia/' + embedia_file + '.h']

        if type_file != '':
            headers.append('../embedia/' + type_file + '.h')

        headers.append('../embedia/debug/embedia_debug.h')
        return headers

    def _generate_data_structure(self, datatype, var_name, conv_datatype, data_converter, data):
        conv_data = data_converter.fit_transform(data.flatten())
        str_data = np.array2string(conv_data, separator=', ', threshold=np.inf)[1:-1]
        shape = data.shape[1:]
        if len(shape) == 3: # tf:channel last => embedia: channel first
            shape = (shape[2], shape[1], shape[0])


        str_dim = ', '.join(map(str, shape))
        return f'{datatype} {var_name} = {{ {str_dim}, ({conv_datatype}[]){{ {str_data} }} }}'

    def _get_measure_function(self, datatype, conv_fn):
        code = '''
typedef struct{
    float acc_error;
    int match;
    int total;
} measures_info_t;
'''
        if datatype == 'data3d_t':
            code+= f'''
float measure_error(data3d_t o_real, data3d_t o_pred, float bnd_error, measures_info_t* info){{
    int x, y, c, pr, pp;
    float error;

    info->total=o_real.channels*o_real.height*o_real.width;
    info->match=0;
    info->acc_error=0;

    for (c=0, pp=0; c<o_real.channels; c++){{
        for (y=0; y<o_real.height; y++){{
            for (x=0; x<o_real.width; x++, pp++){{
               pr = (y*o_real.width+x)*o_real.channels + c;
               printf("%f   %f\\n", {conv_fn("o_real.data[pp]")}, {conv_fn("o_pred.data[pr]")});
               error = fabs({conv_fn("o_real.data[pr]")}-{conv_fn("o_pred.data[pp]")});
               info->acc_error += error;
               if (error <= bnd_error)
                    info->match++;
            }}
        }}
    }}
}}
'''
        elif datatype == 'data2d_t':
            code+= f'''
float measure_error(data2d_t o_real, data2d_t o_pred, float bnd_error, measures_info_t* info){{
    int i;
    float error;

    info->total=o_real.width*o_real.height;
    info->match=0;
    info->acc_error=0;

    for (i=0; i<o_real.width*o_real.height; i++){{
        printf("%f   %f\\n", {conv_fn("o_real.data[i]")}, {conv_fn("o_pred.data[i]")});
        error = fabs({conv_fn("o_real.data[i]")}-{conv_fn("o_pred.data[i]")});
        info->acc_error += error;
        if (error <= bnd_error)
            info->match++;
    }}
}}
'''
        else:
            code+= f'''
float measure_error(data1d_t o_real, data1d_t o_pred, float bnd_error, measures_info_t* info){{
    int i;
    float error;

    info->total=o_real.length;
    info->match=0;
    info->acc_error=0;

    for (i=0; i<o_real.length; i++){{
        printf("%f   %f\\n", {conv_fn("o_real.data[i]")}, {conv_fn("o_pred.data[i]")});
        error = fabs({conv_fn("o_real.data[i]")}-{conv_fn("o_pred.data[i]")});
        info->acc_error += error;
        if (error <= bnd_error)
            info->match++;
    }}
}}
'''
        return code

    def get_filenames(self):
        output_main =  os.path.join(os.path.abspath(self._output_path))
        filenames = [ os.path.join(self.get_project_folder(), 'main.c') ]
        source_folder = os.path.abspath(os.path.join(output_main, 'embedia',  ModelDataType.get_name(self._embedia_type)))

        for c_file in glob.glob(os.path.join(source_folder, '*.c')):
            filenames.append(os.path.abspath(os.path.join(source_folder, c_file)))
        filenames.append(os.path.join(output_main, 'embedia', 'debug', 'embedia_debug.c'))
        return filenames


    def _generate_code_for_layers(self, embedia_layers):
        input_data_type = embedia_layers[0].get_input_data_type()
        output_data_type = embedia_layers[-1].get_output_data_type()
        input_shape = embedia_layers[0].get_input_shape()

        proto_decl = ""
        var_decl = ""
        data_init = ""
        func_impl = ""
        predict = "\n"

        data_layers_input = [{'type': input_data_type, 'var_name': 'input'}, ]
        data_layers_output = [{'type': output_data_type, 'var_name': 'output'}]
        var_output = 'output'
        layer_id = -1
        first_layer = True

        for layer in embedia_layers:
            if layer.layer is None:
                predict += f'    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//\n'
            else:
                layer_id += 1
                predict += f'    //************************ LAYER {layer_id:2d} ***********************//\n'

            if isinstance(layer, DataLayer):
                proto_decl += layer.prototypes_init()   # data init function prototype declaration
                var_decl += layer.var()                 # data variable declaration
                data_init += layer.init()               # variable initialization via data init function
                func_impl += layer.functions_init()     # data init function implementation

            # layer section of predict function
            if not layer.inplace_output:
                input_layer_type = layer.get_input_data_type()
                if data_layers_input[-1]['type'] != input_layer_type:
                    var_input = f'input{len(data_layers_input)}'
                    predict += f'{input_layer_type} {var_input};\n'
                    data_layers_input.append({'type': input_layer_type, 'var_name': var_input})

                if not first_layer:
                    predict += f'    {data_layers_input[-1]["var_name"]} = {data_layers_output[-1]["var_name"]};\n'
                else:
                    first_layer = False

                output_layer_type = layer.get_output_data_type()
                if data_layers_output == [] or data_layers_output[-1]['type'] != output_layer_type:
                    var_output = f'output{len(data_layers_output)}'
                    predict += f'    {output_layer_type} {var_output};\n'
                    data_layers_output.append({'type': output_layer_type, 'var_name': var_output})
            elif first_layer:  # layer section of predict function
                first_layer = False
                predict += f'    {data_layers_output[-1]["var_name"]} = {data_layers_input[-1]["var_name"]};\n'

            param_in = data_layers_input[-1]['var_name']
            param_out = data_layers_output[-1]['var_name']
            predict += f'    {layer.predict(param_in, param_out)}\n\n'

            # Add activation function
            if not isinstance(layer, Activation):
                act_fn = layer.activation_function(var_output)
                if act_fn != '':
                    predict += f'    // Activation layer for {layer.name}\n'
                    predict += f'    {act_fn}\n'

            # debug info
            dbg_fn = layer.debug_function(var_output)
            predict += f'// Debug function for layer {layer.name}\n'
            predict += f'{dbg_fn}\n'

        return (proto_decl, var_decl, data_init, func_impl, predict)


    def _generate_main_code(self, embedia_model, input, output, error_bound):

        test_layer = embedia_model.embedia_layers[-1]
        input_data_type = embedia_model.embedia_layers[0].get_input_data_type()
        output_data_type = embedia_model.embedia_layers[-1].get_output_data_type()
        if embedia_model.is_data_quantized():
            (data_type, data_converter) = embedia_model.get_type_converter(ModelDataType.FLOAT)
        else:
            (data_type, data_converter) = embedia_model.get_type_converter()

        if data_type.startswith('fixed'):
            conv_fn = lambda x : f'FX2FL({x})'
        else:
            conv_fn = lambda x : x

        main_code = '#include <stdlib.h>\n#include <stdio.h>\n#include <math.h>\n'

        for header in self._get_embedia_include_headers():
            main_code += f'#include "{header}"\n'

        (proto_decl, var_decl, data_init, func_impl, predict) = self._generate_code_for_layers(embedia_model.embedia_layers)
        main_code += func_impl+'\n'  # implementation of layer/element intitialization function
        main_code += self._get_measure_function(output_data_type, conv_fn)+'\n' # error measure function

        input_var = self._generate_data_structure(input_data_type, 'input', data_type, data_converter, input)
        output_var = self._generate_data_structure(output_data_type, 'real_output', data_type, data_converter, output)
        # for test
        if self._embedia_type in [ModelDataType.FIXED32, ModelDataType.FIXED16, ModelDataType.FIXED8]:
            compare = 'FX2FL(real_output.data[i]) - FX2FL(output.data[i])'
        else:
            compare = 'real_output.data[i] - output.data[i]'


        main_code += input_var + ';\n\n'
        main_code += output_var + ';\n\n'
        main_code += var_decl+'\n'  # layer/element/module variable declaration
        main_code += output_data_type + ' output;\n\n'  # output var declaration
        main_code += f'''
#define ERROR_BOUND {error_bound}

measures_info_t info; 
  
int main(){{

    {data_init}
    {predict}
    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \\n", info.acc_error);
    printf(" Elem count: %3d \\n", info.total);
    return 0;
}}'''

        return main_code

    def generate(self, model, input, output, error_bound=0.0001):
        # prepare embedia project for export code
        options = ProjectOptions()
        options.data_type = self._embedia_type
        embedia_model = ModelFactory.create_model(model, options)


        main_code = self._generate_main_code(embedia_model, input, output, error_bound)

        self._create_project_folder()
        self._copy_embedia_files()
        self._create_file('main.c', main_code)

        # print(main_code)


