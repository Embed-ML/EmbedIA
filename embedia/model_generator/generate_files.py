import os
import re
import numpy as np
from embedia.model_generator.project_options import (
        ModelDataType,
        ProjectType,
        DebugMode
)
from embedia.utils import file_management
from embedia.utils.c_helper import replace_c_define, CBuilder, ArduinoBuilder
from embedia.model_generator.project_options import BinaryBlockSize
from embedia.core.unimplemented_layer import UnimplementedLayer
from embedia.core.dummy_layer import DummyLayer
from embedia.core.embedia_model import OutputPredictionType
import embedia.utils.file_management as fm


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


def find_source_file(filename, search_paths):
    """
    Encuentra la ubicación real de un archivo fuente en múltiples ubicaciones.

    Args:
        filename: Nombre del archivo a buscar
        search_paths: Lista de directorios donde buscar (en orden de prioridad)

    Returns:
        Ruta completa del archivo encontrado, o None si no se encuentra

    Ejemplo:
        find_source_file('common.h', [src_folder, tmpl_folder])
    """
    for folder in search_paths:
        if not folder or not os.path.exists(folder):
            continue

        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return file_path

    return None


def generate_embedia_library(embedia_model, tmpl_folder, src_folder, dst_folder, ext_h, ext_c, options):

    if options.verbose:
        print("buffer size:", embedia_model.get_buffer_layer_max_size(align=4))

    # 1. Crear DirectiveProcessor con rutas de búsqueda
    processor = fm.DirectiveProcessor(
        embedia_model,
        options,
        search_paths=[src_folder, tmpl_folder]  # ← NUEVO: pasar carpetas de búsqueda
    )

    # 2. Generar contenido de headers (sin cambios)
    includes = []
    defines = {}

    if options.project_type == ProjectType.ARDUINO:
        includes.append('"Arduino.h"')
    else:
        includes.append('<stdlib.h>')

    if options.data_type in [
        ModelDataType.BINARY,
        ModelDataType.BINARY_FIXED32,
        ModelDataType.BINARY_FLOAT16
    ]:
        block_sizes = {
            'Bits8': 8,
            'Bits16': 16,
            'Bits32': 32,
            'Bits64': 64,
        }
        tam_block = block_sizes.get(str(options.tamano_bloque), 64)
        defines['binary_block_size'] = tam_block

    lines = []
    for inc in includes:
        lines.append(f'#include {inc}')

    if includes and defines:
        lines.append('')

    for name, value in defines.items():
        lines.append(f'#define {name} {value}')

    includes_h = '\n'.join(lines) + '\n' if lines else ''

    # 3. Configurar transformaciones por archivo
    transforms = {
        'common.h': {
            'inject_headers': includes_h,
            'update_defines': {'EMBEDIA_MODEL_STORAGE': options.model_storage.qualifier}
        },
        'common.c': {
            'update_defines': {'ALLOC_BUFFER_SZ': embedia_model.get_buffer_layer_max_size(align=4)}
        }
    }

    if options.data_type.is_fixed_point and options.fixed_precision is not None:
        transforms['fixed.h'] = {
            'update_defines': {
                'FIX_FRC_SZ': options.fixed_precision
            }
        }
    
    # agregar redefiniciones de macros definidas en archivos a incluir
    for (hfile, cfile) in embedia_model.required_files:
        # hfile y cfile son EmbediaFile o None
        if hfile and hasattr(hfile, 'defines') and hfile.defines:
            transforms[str(hfile)] = {
                'update_defines': hfile.defines
            }
        if cfile and hasattr(cfile, 'defines') and cfile.defines:
            transforms[str(cfile)] = {
                'update_defines': cfile.defines
            }

    # 4. Procesar archivos requeridos
    required_files = embedia_model.required_files
    embedia_files = []

    for (header_file, code_file) in required_files:
        for element in [header_file, code_file]:
            if not element:  # es None
                continue

            filename = str(element)  # Usar str() para obtener el nombre

            if filename.endswith('.c'):
                new_name = filename.replace('.c', ext_c)
            elif filename.endswith('.h'):
                new_name = filename.replace('.h', ext_h)
            else:
                new_name = filename

            # Buscar archivo en tmpl_folder o src_folder
            src_path = find_source_file(filename, [tmpl_folder, src_folder])
            dst_path = os.path.join(dst_folder, new_name)

            transform = transforms.get(filename, {})
            processor.process_file_full(
                src_path,
                dst_path,
                inject_headers=transform.get('inject_headers'),
                update_defines=transform.get('update_defines')
            )

            embedia_files.append(new_name)

    return embedia_files



def get_input_const(input_shape):
    if len(input_shape) == 3:
        return {'INPUT_CHANNELS': input_shape[2], 'INPUT_WIDTH': input_shape[1], 'INPUT_HEIGHT': input_shape[0]}
    elif len(input_shape) == 2:
        # for 2D channels and height are alias
        return {'INPUT_CHANNELS': input_shape[0], 'INPUT_WIDTH': input_shape[1]}
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
    full_quant = options.data_type == ModelDataType.FULL_QUANT8

    include_files = set([model_filename])
    if options.debug_mode != DebugMode.DISCARD:
        if full_quant:
            include_files.add('embedia_debug_quant')
        else:
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
                # Usar str() para obtener el nombre, funciona tanto con EmbediaFile como con strings
                filename = str(header_file)
                include_files.add(filename[0:-2])

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

                input_layer_type = layer.input_data_type
                output_layer_type = layer.output_data_type

                # layer section of predict function
                if not layer.inplace_output:

                    if data_layers_input[-1]['type'] != input_layer_type:
                        var_input = f'input{len(data_layers_input)}'
                        predict_fn += f'{input_layer_type} {var_input};\n'
                        data_layers_input.append({'type': input_layer_type, 'var_name': var_input})

                    if not first_layer:
                        predict_fn += f'{data_layers_input[-1]["var_name"]} = {data_layers_output[-1]["var_name"]};\n'
                    else:
                        first_layer = False

                    if data_layers_output == [] or data_layers_output[-1]['type'] != output_layer_type:
                        var_output = f'output{len(data_layers_output)}'
                        predict_fn += f'{output_layer_type} {var_output};\n'
                        data_layers_output.append({'type': output_layer_type, 'var_name': var_output})

                elif first_layer: # first layer is inplace so output must exist before use
                    first_layer = False
                    var_input = 'input'
                    var_output = f'output{len(data_layers_output)}'
                    n_dims = output_data_type[-4:-2] #1d, 2d or 3d
                    predict_fn += '// copy input because first layer is inplace\n'
                    predict_fn += f'{output_layer_type} {var_output};\n'
                    predict_fn += f'copy_data_{n_dims}(&{var_input}, &{var_output});\n'
                    data_layers_output.append({'type': output_layer_type, 'var_name': var_output})


                param_in = data_layers_input[-1]['var_name']
                param_out = data_layers_output[-1]['var_name']
                predict_fn += f'{layer.invoke(param_in, param_out)}\n'

                # Add debug function if is enabled
                if options.debug_mode != DebugMode.DISCARD:
                    dbg_fn = layer.debug_function(var_output, full_quant=full_quant)
                    predict_fn += f'// Debug function for layer {layer.name}\n'
                    predict_fn += f'{dbg_fn}\n'
            else:
                # message of unimplemented layer
                predict_fn += '// ' + layer.message + '\n'

    #if data_layers_output[-1]["var_name"] != 'output':
    #    predict_fn += f'   output = {data_layers_output[-1]["var_name"]};\n'

    # indent code
    predict_fn = indent(predict_fn)
    # improve code in order to include the correct model funcion
    predict_class = ''
    if output_data_type == 'data1d_t':
        output_pred_type = model.output_prediction_type
        if output_pred_type == OutputPredictionType.BINARY_OUTPUT:
            predict_class = 'return results->data[0] >= 0.5;'
        elif output_pred_type == OutputPredictionType.CLASS_PROBABILITIES:
            predict_class = 'return argmax(results->data, results->length);'
        elif output_pred_type == OutputPredictionType.DIRECT_CLASS_ID:
            predict_class = 'return results->data[0];'
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


def generate_embedia_main(embedia_model, src_folder, dst_embedia_folder, model_name, options, strategy):
    embedia_layers = embedia_model._embedia_layers

    src_c = os.path.join(src_folder, 'main/main_')

    # Use strategy for includes and coder
    coder = strategy.get_coder()
    includes_c = strategy.get_includes()
    baud_rate = strategy.get_baud_rate(options)
    
    if options.project_type == ProjectType.ARDUINO:
        src_c += "arduino.c"
    else:
        src_c += "c.c"

    # for basic types of embedia
    filename = os.path.join(dst_embedia_folder, 'neural_net.h')
    includes_c += f'#include "{filename}"\n'

    filename = os.path.join(dst_embedia_folder, model_name+'.h')
    includes_c += f'#include "{filename}"\n'

    example_var_name = 'sample_data'

    # The code generated is a part of the main function => start indented
    coder.inc()

    if options.example_data is not None:
        filename = os.path.join(dst_embedia_folder, 'example_file.h')
        includes_c += f'#include "{filename}"\n'

    coder.append('// model initialization')
    coder.append('model_init();')

    # prepare data for model input and output
    input_data_type = embedia_layers[0].input_data_type
    output_data_type = embedia_layers[-1].output_data_type

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado no se usa igual por ahora

    if options.data_type == ModelDataType.FLOAT or options.data_type == ModelDataType.BINARY:
        model_data_type = 'float'
    elif options.data_type in [ModelDataType.QUANT8, ModelDataType.FULL_QUANT8]:
        model_data_type = 'quant8'
    else:
        model_data_type = 'fixed'

    input_const = get_input_const(embedia_layers[0].input_shape)

    # Detectar si es data2d_t y formatear apropiadamente
    if embedia_layers[0].input_shape == 2 and len(input_const) == 2:
        # Para data2d_t con 2 campos (channels/height + width)
        values = list(input_const.values())
        # El primer valor va dentro de la union, el segundo es width
        input_initializer = f'{{ {{ {values[0]} }}, {values[1]}, NULL }}'
    else:
        # Formato normal
        input_dim = ', '.join([f'{v}' for v in input_const.values()])
        input_initializer = f'{{ {input_dim}, NULL }}'

    input_data = f'''{input_data_type} input = {input_initializer};\n'''

    output_data = f'''{output_data_type} results;\n'''

    coder.append(f'''
// make model prediction
// uncomment corresponding code

// int prediction = model_predict_class(input, &results);

// print predicted class id''')


    if options.example_data is not None:
        coder.append('int i, ok=0, prediction;')
        coder.printf('example_file.h tests\\n')
        coder.printf('Error | Cls | Pred \\n')
        coder.printf('------|-----|------\\n')
        with coder.bgn('for (i=0; i<TEST_SAMPLES; i++) {'):
            coder.append('input.data = sample_data[i];')
            coder.append('prediction = model_predict_class(input, &results);')
            with coder.bgn('if (prediction == sample_data_ids[i][0]) {'):
                coder.append('ok++;')
                coder.printf('      |  %2d |  %2d  \\n', 'sample_data_ids[i][0]', 'prediction')
            with coder.bgn('else {'):
                coder.printf('   X  |  %2d |  %2d  \\n', 'sample_data_ids[i][0]', 'prediction')
        coder.printf('\\n%d correct out of %d (Accuracy: %.2f%%)\\n', 'ok', 'TEST_SAMPLES', '(100.0 * ok)/TEST_SAMPLES')

    main_code = coder.get_code()

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


def generate_embedia_debug(src_dbg_folder, dst_folder, options, strategy, ext_h='.h', ext_c='.c'):
    # add debug mode macro to header file

    if options.data_type == ModelDataType.FULL_QUANT8:
        debug_filename = 'embedia_debug_quant'
    else:
        debug_filename = 'embedia_debug'

    content = file_management.read_from_file(os.path.join(src_dbg_folder, f'{debug_filename}{ext_h}'))
    # add include
    content = content.format(EMBEDIA_DEBUG='#define EMBEDIA_DEBUG %d\n' % options.debug_mode)
    file_management.save_to_file(os.path.join(dst_folder, f'{debug_filename}{ext_h}'), ''.join(content))
    
    # Use strategy to get debug file names
    def_header, impl_file = strategy.get_debug_files(debug_filename)
    
    # copy additional debug file
    file_management.copy(os.path.join(src_dbg_folder, def_header),
                os.path.join(dst_folder, f'embedia_debug_def{ext_h}'))
    
    # copy implementation file with correct extension
    file_management.copy(os.path.join(src_dbg_folder, f'{debug_filename}.c'),
                os.path.join(dst_folder, f'{impl_file[:-2]}{ext_c}'))


def data_to_array_str(data, macro_converter=None, clip=120):
    if macro_converter is None:
        macro_converter = lambda x:x
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

    if options.data_type == ModelDataType.QUANT8:
        (data_type, data_converter) = embedia_model.get_type_converter(ModelDataType.FIXED16)
    else:
        (data_type, data_converter) = embedia_model.get_type_converter()

    src_h = os.path.join(src_folder, 'main/example_file.h')
    smp = options.example_data
    ids = options.example_ids

    #print("smp", smp)

    if smp.shape[0] != ids.shape[0]:
        raise Exception("The number of examples does not match the number of classes")
    if not isinstance(smp, np.ndarray):
        smp = np.array(smp)
    if not isinstance(ids, np.ndarray):
        ids = np.array(ids)
    if len(ids.shape) == 1:
        ids = ids.reshape((-1,1))
    ids = ids.astype(int)


    # generate array of samples
    data_samples = ''
    data_converter.fit(smp)
    if options.data_type == ModelDataType.FULL_QUANT8:
        data_samples_quant =f'''
const qparam_t {var_name}_qp = {{
    (int32_t) ({data_converter.scale}*Q_SCALE), // Escala
    {data_converter.zero_pt} // Punto cero
}};'''
    else:
        data_samples_quant =''

    for i in range(len(smp)):
        data = smp[i].flatten()
        #new_data = data_converter.fit_transform(data)
        new_data = data_converter.transform(data)
        #id = int(ids[i])
        comma = ',' if i > 0 else ' '
        data_samples += f'''#if (FST_TEST_SAMPLE <= {i}) && ({i} <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != {i})
    ,
    #endif
    {{ {data_to_array_str(new_data)} }}
#endif
'''
    data_predict = ''
    for i in range(len(ids)):

        # id = int(ids[i])
        comma = ',' if i > 0 else ' '
        data_predict += f'''#if (FST_TEST_SAMPLE <= {i}) && ({i} <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != {i})
    ,
    #endif
        {{ {data_to_array_str(ids[i])} }}
#endif
'''
    smp_size = '*'.join(map(str, smp.shape[1:]))
    examples = f'''
// The sample array below may contain up to {smp.shape[0]-1} elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, {smp.shape[0]-1}] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE {smp.shape[0]-1}
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)

{data_samples_quant}

static {data_type} {var_name}[][{smp_size}]= {{
{data_samples}
}};

static int {var_name}_ids[][{smp_size}]= {{
{data_predict}
}};

'''

    content = file_management.read_from_file(src_h).format(examples=examples)

    return content
