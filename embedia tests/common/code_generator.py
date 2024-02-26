from enum import Enum
import os
import glob
import shutil
import numpy as np

from embedia.layers.model import Model as EmbediaModel
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

    def _copy_embedia_files(self):
        source_path = os.path.abspath(os.path.join(self._embedia_path, 'libraries'))
        dest_path = os.path.abspath(self._output_path)

        for datatype in [ModelDataType.FLOAT, ModelDataType.FIXED32, ModelDataType.FIXED16, ModelDataType.FIXED8,
                         ModelDataType.QUANT8]:
            folder = ModelDataType.get_name(datatype)
            src_folder = os.path.join(source_path, folder)
            dst_folder = os.path.join(dest_path, 'embedia', folder)
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

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
        if datatype == 'data3d_t':
            return f'''
float measure_error(data3d_t o_real, data3d_t o_pred, float err){{
    int x, y, c, pr, pp, match;
    for (match=0,c=0, pp=0; c<o_real.channels; c++){{
        for (y=0; y<o_real.height; y++){{
            for (x=0; x<o_real.width; x++, pp++){{
               pr = (y*o_real.width+x)*o_real.channels + c;
               printf("%f   %f\\n", {conv_fn("o_real.data[pp]")}, {conv_fn("o_pred.data[pr]")});
               if (fabs({conv_fn("o_real.data[pr]")}-{conv_fn("o_pred.data[pp]")}) <= err)
                    match++;
            }}
        }}
    }}
    return 100.0*match/(o_real.channels*o_real.height*o_real.width);
}}'''
        elif datatype == 'data2d_t':
            return f'''
            float measure_error(data2d_t o_real, data2d_t o_pred, float err){{
                int i, match;
                for (match=0,i=0; i<o_real.width*o_real.height; i++){{
                    printf("%f   %f\\n", {conv_fn("o_real.data[i]")}, {conv_fn("o_pred.data[i]")});
                    if (fabs({conv_fn("o_real.data[i]")}-{conv_fn("o_pred.data[i]")}) <= err){{
                        match++;
                    }}
                }}
                return 100.0*match/(o_real.width*o_real.height);
            }}
'''
        else:
            return f'''
float measure_error(data1d_t o_real, data1d_t o_pred, float err){{
    int i, match;
    for (match=0,i=0; i<o_real.length; i++){{
        printf("%f   %f\\n", {conv_fn("o_real.data[i]")}, {conv_fn("o_pred.data[i]")});
        if (fabs({conv_fn("o_real.data[i]")}-{conv_fn("o_pred.data[i]")}) <= err){{
            match++;
        }}
    }}
    return 100.0*match/o_real.length;
}}
'''

    def get_filenames(self):
        output_main =  os.path.join(os.path.abspath(self._output_path))
        filenames = [ os.path.join(self.get_project_folder(), 'main.c') ]
        source_folder = os.path.abspath(os.path.join(output_main, 'embedia',  ModelDataType.get_name(self._embedia_type)))

        for c_file in glob.glob(os.path.join(source_folder, '*.c')):
            filenames.append(os.path.abspath(os.path.join(source_folder, c_file)))

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

        return (proto_decl, var_decl, data_init, func_impl, predict)

    # def _generate_main_code_old_casi_ok(self, embedia_model, input, output, error_bound):
    #
    #     test_layer = embedia_model.embedia_layers[-1]
    #     input_data_type = test_layer.get_input_data_type()
    #     output_data_type = test_layer.get_output_data_type()
    #     if embedia_model.is_data_quantized():
    #         (data_type, data_converter) = embedia_model.get_type_converter(ModelDataType.FLOAT)
    #     else:
    #         (data_type, data_converter) = embedia_model.get_type_converter()
    #
    #     if data_type.startswith('fixed'):
    #         conv_fn = lambda x: f'FX2FL({x})'
    #     else:
    #         conv_fn = lambda x: x
    #
    #     main_code = '#include <stdlib.h>\n#include <stdio.h>\n#include <math.h>\n'
    #     for header in self._get_embedia_include_headers():
    #         main_code += f'#include "{header}"\n'
    #
    #     main_code += test_layer.functions_init() + '\n'  # implementation of layer/element intitialization function
    #     main_code += self._get_measure_function(output_data_type, conv_fn) + '\n'  # error measure function
    #     main_code += test_layer.var()  # layer/element variable declaration
    #
    #     input_var = self._generate_data_structure(input_data_type, 'input', data_type, data_converter, input) + '\n'
    #     output_var = self._generate_data_structure(output_data_type, 'real_output', data_type, data_converter,
    #                                                output) + '\n'
    #     # for test
    #     if self._embedia_type in [ModelDataType.FIXED32, ModelDataType.FIXED16, ModelDataType.FIXED8]:
    #         compare = 'FX2FL(real_output.data[i]) - FX2FL(output.data[i])'
    #     else:
    #         compare = 'real_output.data[i] - output.data[i]'
    #
    #     # total of output values, regards the shape
    #     output_count = np.prod(output.shape)
    #
    #     main_code += input_var + ';\n'
    #     main_code += output_var + ';\n'
    #     main_code += output_data_type + ' output;\n\n'
    #     main_code += f'# define ERROR_BOUND {error_bound}\n'
    #     main_code += 'int main(){\n\n'  # main code start
    #     main_code += test_layer.init() + '\n'  # call to layer/element initizalization function
    #     main_code += '    ' + test_layer.predict('input', 'output') + '\n'
    #
    #     main_code += f'''
    #     printf("Test result: %6.3f %%\\n", measure_error(real_output, output, ERROR_BOUND));
    #
    # '''
    #     main_code += '    return 0;\n}'
    #
    #     return main_code

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
        main_code += f'# define ERROR_BOUND {error_bound}\n\n'
        main_code += 'int main(){\n\n'  # main code start
        main_code += data_init # call to layer/element/module initizalization function
        main_code += '    ' + predict

        main_code += f'    printf("Test result: %6.3f %%\\n", measure_error(real_output, output, ERROR_BOUND));\n\n'
        main_code += '    return 0;\n}'

        return main_code

    def generate(self, model, input, output, error_bound=0.0001):
        # prepare embedia project for export code
        options = ProjectOptions()
        options.data_type = self._embedia_type
        embedia_model = EmbediaModel(options)
        embedia_model.set_model(model)

        main_code = self._generate_main_code(embedia_model, input, output, error_bound)

        self._create_project_folder()
        self._copy_embedia_files()
        self._create_file('main.c', main_code)

        # print(main_code)


