import os
import re
import numpy as np
from embedia.model_generator.project_options import (
        ModelDataType,
        ProjectType,
        DebugMode
)
from embedia.utils import file_management
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.core.unimplemented_layer import UnimplementedLayer
from embedia.core.dummy_layer import DummyLayer

def multi_replace(adict, text):
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys())))

    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)


def indent(multi_ln_code, level=1, spaces=4):

    # remove the whitespaces at the end of each line
    code = re.sub(r'[ \t]+(?=\n)', '', multi_ln_code)

    # check if the code has a trailing new line
    has_nl = code.endswith("\n")

    return re.sub("^", level*spaces*' ', code, flags=re.MULTILINE)


def generate_embedia_library(embedia_model, src_folder, dst_folder, ext_h, ext_c, options):

    # files to add "#include"
    update_include_files = ['common.h']

    filenames = os.listdir(src_folder)
    required_files = embedia_model.required_files

    embedia_files = []

    for (header_file, code_file) in required_files:
        # check if project's files exists
        if header_file is not None:
            if header_file not in filenames:
                raise FileNotFoundError(f'Missing file: {header_file} in {src_folder}')
            embedia_files.append(header_file)
        if code_file is not None:
            if code_file not in filenames:
                raise FileNotFoundError(f'Missing file: {code_file} in {src_folder}')
            embedia_files.append(code_file)

    # Prepare includes for files
    includes_h = ''
    if options.data_type == ModelDataType.BINARY or options.data_type == ModelDataType.BINARY_FIXED32 or options.data_type == ModelDataType.BINARY_FLOAT16:

        if options.tamano_bloque == BinaryBlockSize.Bits8:
            tam_block = 8
        elif options.tamano_bloque == BinaryBlockSize.Bits16:
            tam_block = 16
        elif options.tamano_bloque == BinaryBlockSize.Bits32:
            tam_block = 32
        else:
            tam_block = 64

        if options.project_type == ProjectType.ARDUINO:
            includes_h = '#include "Arduino.h"\n'
            includes_h += f'\n#define binary_block_size {tam_block}\n'
        else:
            includes_h = '#include <stdlib.h>\n'
            includes_h += f'\n#define binary_block_size {tam_block}\n'
    else:
        if options.project_type == ProjectType.ARDUINO:
            includes_h = '#include "Arduino.h"\n'
        else:
            includes_h = '#include <stdlib.h>\n'

    for i, filename in enumerate(embedia_files):
        if filename.endswith('.c'):
            new_name = filename.replace('.c', ext_c)
        elif filename.endswith('.h'):
            new_name = filename.replace('.h', ext_h)
        else:
            new_name = filename

        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, new_name)
        if filename in update_include_files:
            content = file_management.read_from_file(src_file)
            content = multi_replace({'{includes}': includes_h}, content)
            file_management.save_to_file(dst_file, content)
        else:
            file_management.copy(src_file, dst_file)

        # update with new filename
        embedia_files[i] = new_name

    return embedia_files


# Warning TO DO: check width vs height order
def get_input_const(input_shape):
    if len(input_shape) == 3:
        return {'INPUT_CHANNELS': input_shape[2], 'INPUT_WIDTH': input_shape[1], 'INPUT_HEIGHT': input_shape[0]}
    elif len(input_shape) == 2:
        return {'INPUT_WIDTH': input_shape[1], 'INPUT_HEIGHT': input_shape[0]}
    elif len(input_shape) == 1:
        return {'INPUT_LENGTH': input_shape[0]}

    return None


def generate_embedia_model(model, src_folder, dst_folder, ext_h, ext_c, model_name, model_info, options):
    def format_model_name(model_name):
        model_name = model_name.lower()
        if not model_name.endswith('model'):
            model_name += '_model'
        return model_name

    embedia_layers = model.embedia_layers

    model_filename = format_model_name(model_name)

    src_h = os.path.join(src_folder, 'model/model.h')
    src_c = os.path.join(src_folder, 'model/model.c')

    include_files = set([model_filename])
    if options.debug_mode != DebugMode.DISCARD:
        include_files.add('embedia_debug')

    model_name_h = f'_{model_filename.upper()}_H'
    # macros_first_shape = embedia_layers[0].get_macros_first_shape()
    input_data_type = embedia_layers[0].input_data_type
    output_data_type = embedia_layers[-1].output_data_type
    input_shape = embedia_layers[0].input_shape

    # prepare input dimension constant
    input_dict = get_input_const(input_shape)
    input_const = ""
    total_size = 1
    for k in input_dict:
        total_size *= input_dict[k]
        input_const += f'#define {k} {input_dict[k]}\n'
    input_const += f'\n#define INPUT_SIZE {total_size}\n'

    prototypes_init = ""
    var_decl = ""
    data_init = ""
    func_impl = ""
    predict_fn = "prepare_buffers();\n"

    data_layers_input = [{'type': input_data_type, 'var_name': 'input'}, ]
    data_layers_output = []
    layer_id = -1
    first_layer = True

    for layer in embedia_layers:
        # includes files of function prototype and implementation
        files_list = layer.required_files
        for (header_file, code_file) in files_list:
            if header_file is not None:
                include_files.add(header_file[0:-2])

        if layer.wrapper is None:
            predict_fn += f'\n//<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//'
        else:
            layer_id += 1
            predict_fn += f'\n//******************** LAYER {layer_id} *******************//'

        predict_fn += f'\n// Layer name: {layer.name}\n'

        if not isinstance(layer, DummyLayer):
            implemented_layer = not isinstance(layer, UnimplementedLayer)

            if implemented_layer:
                # Initialization
                if layer.use_data_structure:
                    prototypes_init += layer.function_prototype      # data init function prototype declaration
                    var_decl += layer.variable_declaration      # data variable declaration
                    data_init += layer.variable_initialization  # variable initialization via data init function
                    func_impl += layer.function_implementation  # data init function implementation

                # layer section of predict function
                if not layer.inplace_output:

                    input_layer_type = layer.input_data_type
                    if data_layers_input[-1]['type'] != input_layer_type:
                        var_input = f'input{len(data_layers_input)}'
                        predict_fn += f'{input_layer_type} {var_input};\n'
                        data_layers_input.append({'type': input_layer_type, 'var_name': var_input})

                    if not first_layer:
                        predict_fn += f'{data_layers_input[-1]["var_name"]} = {data_layers_output[-1]["var_name"]};\n'
                    else:
                        first_layer = False

                    output_layer_type = layer.output_data_type
                    if data_layers_output == [] or data_layers_output[-1]['type'] != output_layer_type:
                        var_output = f'output{len(data_layers_output)}'
                        predict_fn += f'{output_layer_type} {var_output};\n'
                        data_layers_output.append({'type': output_layer_type, 'var_name': var_output})

                param_in = data_layers_input[-1]['var_name']
                param_out = data_layers_output[-1]['var_name']
                predict_fn += f'{layer.invoke(param_in, param_out)}\n'

                # Add debug function if is enabled
                if options.debug_mode != DebugMode.DISCARD:
                    dbg_fn = layer.debug_function(var_output)
                    predict_fn += f'// Debug function for layer {layer.name}\n'
                    predict_fn += f'{dbg_fn}\n'
            else:
                # message of unimplemented layer
                predict_fn += '// ' + layer.message + '\n'

    # indent code
    predict_fn = indent(predict_fn)
    # improve code in order to include the correct model funcion
    if output_data_type == 'data1d_t':
        n_classes = model.identify_target_classes()  # determine model classes, 0=regression, 1=binary, >1=multiclass
        if n_classes == 1:
            predict_class = 'return results->data[0] >= 0.5;'
        else:
            predict_class = 'return argmax(*results);'
    else:
        predict_class = '''//TO DO: argmax with data2d_t and data3d_t
    return -1; '''

    # prepare include files
    includes = ''
    for filename in include_files:
        includes += f'#include "{filename}.h"\n'

    text_model_h = file_management.read_from_file(src_h).format(
            model_name_h=model_name_h,
            model_info=model_info,
            input_const=input_const,
            input_data_type=input_data_type,
            output_data_type=output_data_type
        )

    text_model_c = file_management.read_from_file(src_c).format(
            includes=includes,
            filename=model_filename,
            prototypes_init=prototypes_init,
            var=var_decl,
            init=data_init,
            predict=predict_fn,
            predict_class=predict_class,
            functions_init=func_impl,
            input_data_type=input_data_type,
            output_data_type=output_data_type,
            output_name=var_output
        )

    file_management.save_to_file(os.path.join(dst_folder, model_filename + ext_h), text_model_h)
    file_management.save_to_file(os.path.join(dst_folder, model_filename + ext_c), text_model_c)
    return (text_model_h, text_model_c, model_filename)


def generate_embedia_main(embedia_model, src_folder, dst_embedia_folder, model_name, options):
    embedia_layers = embedia_model._embedia_layers

    src_c = os.path.join(src_folder, 'main/main_')

    # Prepare includes
    if options.project_type == ProjectType.ARDUINO:
        src_c += "arduino.c"
        includes_c = '#include "Arduino.h"\n'
        baud_rate = str(options.baud_rate)
    else:
        src_c += "c.c"
        includes_c = '#include <stdio.h>\n'
        baud_rate = "\n"

    # for basic types of embedia
    filename = os.path.join(dst_embedia_folder, 'neural_net.h')
    includes_c += f'#include "{filename}"\n'

    filename = os.path.join(dst_embedia_folder, model_name+'.h')
    includes_c += f'#include "{filename}"\n'

    example_var_name = 'sample_data'
    main_code = ''

    if options.example_data is not None:
        filename = os.path.join(dst_embedia_folder, 'example_file.h')
        includes_c += f'#include "{filename}"\n'
        main_code += f'''
    // sample intitialization
    input.data = {example_var_name};
'''

    main_code += '''
    // model initialization
    model_init();
'''

    # prepare data for model input and output
    input_data_type = embedia_layers[0].input_data_type
    output_data_type = embedia_layers[-1].output_data_type

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado no se usa igual por ahora

    if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
        model_data_type = 'float'
    elif options.data_type == ModelDataType.QUANT8:
        model_data_type = 'quant8'
    else:
        model_data_type = 'fixed'

    input_const = get_input_const(embedia_layers[0].input_shape)
    input_dim = ''
    for k in input_const:
        input_dim += f'{k}, '

    input_data = f'''{input_data_type} input = {{ {input_dim} NULL}};\n'''
    output_data = f'''{output_data_type} results;\n'''

    main_code += '''

    // make model prediction
    // uncomment corresponding code

    int prediction = model_predict_class(input, &results);

    // print predicted class id'''

    if options.project_type == ProjectType.ARDUINO:
        main_code += '''
    Serial.print("Prediction class id: ");
    Serial.println(prediction);
'''
        if options.example_data is not None:
            main_code += '''
    Serial.print("   Example class id: ");
    Serial.println(sample_data_id);
'''
    else:
        main_code += '''
    printf("Prediction class id: %d\\n", prediction);
'''
        if options.example_data is not None:
            main_code += '''
    printf("   Example class id: %d\\n", sample_data_id);
'''

    main_code += '''
    /*

    model_predict(input, &results);

    printf("prediccion: %.5f", results.data[0]);

    */

'''

    c = file_management.read_from_file(src_c).format(includes=includes_c,
                                                     input_data=input_data,
                                                     output_data=output_data,
                                                     baud_rate=baud_rate,
                                                     main_code=main_code)

    # load and generate data example if it corresponds
    if options.example_data is not None:
        h = generate_examples(src_folder, example_var_name, options, embedia_model)
    else:
        h = None

    return (h, c)


def generate_embedia_debug(src_dbg_folder, dst_folder, options):
    # add debug mode macro to header file
    content = file_management.read_from_file(os.path.join(src_dbg_folder, 'embedia_debug.h'))
    # add include
    content = content.format(EMBEDIA_DEBUG='#define EMBEDIA_DEBUG %d\n' % options.debug_mode)
    file_management.save_to_file(os.path.join(dst_folder, 'embedia_debug.h'), ''.join(content))
    # copy aditional debug file
    if options.project_type == ProjectType.ARDUINO:
        file_management.copy(os.path.join(src_dbg_folder, 'embedia_debug_def_arduino.h'),
                    os.path.join(dst_folder, 'embedia_debug_def.h'))
        # copy implementation file
        file_management.copy(os.path.join(src_dbg_folder, 'embedia_debug.c'),
                    os.path.join(dst_folder, 'embedia_debug.cpp'))
    else:
        file_management.copy(os.path.join(src_dbg_folder, 'embedia_debug_def_c.h'),
                    os.path.join(dst_folder, 'embedia_debug_def.h'))
        # copy implementation file
        file_management.copy(os.path.join(src_dbg_folder, 'embedia_debug.c'),
                    os.path.join(dst_folder, 'embedia_debug.c'))

def generate_codeblock_project(project_name, files, src_folder, _dst_embedia_folder_name):

    embedia_output_folder = _dst_embedia_folder_name
    included_files = ''
    for filename in files:
        if filename[-2:].lower() == '.c':
            if filename == 'main.c':
                folder_filename = filename
            else:
                folder_filename = os.path.join(embedia_output_folder, filename)
            included_files += f'''
        <Unit filename="{folder_filename}">
            <Option compilerVar="CC" />
        </Unit>'''
        elif filename[-2:].lower() == '.h':
            folder_filename = os.path.join(embedia_output_folder, filename)
            included_files += f'''
        <Unit filename="{folder_filename}" />'''

    src_cbp = os.path.join(src_folder, 'main/codeblock_project.cbp')
    content = file_management.read_from_file(src_cbp)

    content = multi_replace({'{project_name}': project_name, '{included_files}': included_files}, content)

    return content


def data_to_array_str(data, macro_converter, clip=120):
    output = ''
    cline = '  '
    for i in data.flatten():
        cline += macro_converter(str(i)) + ', '
        if len(cline) > clip:
            output += cline + '\n'
            cline = '  '
    output += cline
    return output[:-2]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
def generate_examples(src_folder, var_name, options, embedia_model):


    # if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
    #     def conv(s):
    #         return s
    #     data_type = 'float'
    # elif options.data_type == ModelDataType.BINARY_FLOAT16:
    #     def conv(s):
    #         return f"half({s})"
    #     data_type = 'half'
    # else:
    #     def conv(s):
    #         return f"FL2FX({s})"
    #     data_type = 'fixed'

    if embedia_model.is_data_quantized:
        (data_type, data_converter) = embedia_model.get_type_converter(ModelDataType.FLOAT)
    else:
        (data_type, data_converter) = embedia_model.get_type_converter()

    conv = lambda x: x

    src_h = os.path.join(src_folder, 'main/example_file.h')
    smp = options.example_data
    ids = options.example_ids

    if smp.shape[0] != ids.shape[0]:
        raise Exception("The number of examples does not match the number of classes")

    examples = f'''
#define MAX_SAMPLE {smp.shape[0]-1}

#define SELECT_SAMPLE 0

'''
    if not isinstance(smp, np.ndarray):
        smp = np.array(smp)
    for i in range(len(smp)):
        data = smp[i].flatten()
        new_data = data_converter.fit_transform(data)

        id = int(ids[i])
        examples += f'''#if SELECT_SAMPLE == {i}
        
uint16_t {var_name}_id = {id};

static {data_type} {var_name}[]= {{
{data_to_array_str(new_data, conv)}
}};

#endif
'''

    content = file_management.read_from_file(src_h).format(examples=examples)

    return content
