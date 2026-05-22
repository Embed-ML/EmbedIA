import os
import shutil
import warnings

from embedia.core import embedia_model
from embedia.core.model_factory import ModelFactory, DummyModel
from embedia.model_generator.project_options import (
        ModelMicro,
        ModelDataType,
        ProjectType,
        ProjectFiles,
        DebugMode
    )

from embedia.model_generator.generate_files import (
        generate_embedia_library,
        generate_embedia_model,
        generate_embedia_main,
        generate_embedia_debug
    )

from embedia.model_generator.project_strategy import get_project_strategy

from embedia.utils import file_management, diagnostics

from prettytable import PrettyTable

def format_model_name(model):
    model_name = model.name.lower()
    if not model_name.endswith('model'):
        model_name += '_model'

    return model_name


class ProjectGenerator:

    def __init__(self, options):

        self._options = options

        self._root_folder = None
        self._src_lib_folder = None
        self._src_datatype_folder = None
        self._src_dbg_folder = None

        self._dst_folder = None
        self._dst_embedia_folder = None
        self._embedia_model = None
        self._strategy = get_project_strategy(options.project_type)
        self._dst_embedia_folder_name = self._strategy.get_embedia_folder_name(options)

        if options.embedia_folder is None or options.embedia_folder == '':
            self.set_embedia_folder('embedia/')
        else:
            self.set_embedia_folder(options.embedia_folder)

        if options.project_type==ProjectType.C and options.data_type == ModelDataType.BINARY_FLOAT16:
            raise ValueError("FLOAT16 is not compatible with C, only with C++ and Arduino!!")
        if options.project_type==ProjectType.CODEBLOCK and options.data_type == ModelDataType.BINARY_FLOAT16:
            raise ValueError("FLOAT16 is not compatible with CodeBlocks, only with C++ and Arduino!!")


    def set_embedia_folder(self, folder):
        if folder[-1] != '/':
            folder += '/'
        self._root_folder = os.path.abspath(folder) + '/'
        self._src_lib_folder = self._root_folder + 'libraries/'
        self._src_mcu_folder = self._src_lib_folder + 'mcu/'
        self._src_template_folder = self._src_mcu_folder + 'template/'
        self._src_datatype_folder = self._src_lib_folder + self._datatype_subfolder(self._options.data_type)
        self._src_dbg_folder = self._src_lib_folder + 'debug/'

    def check_options(self):

        if not os.path.exists(self._src_datatype_folder):
            txt = f'{self._options.data_type.name} not implemented for {self._options.micro.name} micro. Falling back to GENERIC micro export.'
            diagnostics.warn(txt)
            self._options.micro = ModelMicro.GENERIC

        if self._options.embedia_folder is None or self._options.embedia_folder == '':
            self.set_embedia_folder('embedia/')
        else:
            self.set_embedia_folder(self._options.embedia_folder)

    def create_project(self, output_folder, project_name, model, options):

        # check existing implementation for micro+datatype
        self.check_options()

        if model is None:
            model = DummyModel('No Model')

        embedia_model = ModelFactory.create_model(model, options)

        self._embedia_model = embedia_model


        # prepare folders and extension of files to copy/create
        self._prepare_folders(output_folder, project_name, options)

        c_ext, h_ext = self._get_files_extension()


        # print layers memory size
        model_info = self.build_model_info(embedia_model)
        if options.verbose:
            print(model_info)


        # copy library files
        if ProjectFiles.LIBRARY in options.files:
            #embedia_files = generate_embedia_library(embedia_model, self._src_datatype_folder, self._dst_folder, h_ext, c_ext, options)
            embedia_files = generate_embedia_library(
                embedia_model,
                self._src_template_folder,
                self._src_datatype_folder,
                self._dst_embedia_folder,
                h_ext,
                c_ext,
                options)



        # create model files
        if ProjectFiles.MODEL in options.files:
            (text_model_h, text_model_c, model_name) = generate_embedia_model(embedia_model, self._src_lib_folder, self._dst_embedia_folder, h_ext, c_ext, model.name, model_info, options)

        # copy debug file
        if options.debug_mode != DebugMode.DISCARD:
            generate_embedia_debug(self._src_dbg_folder, self._dst_embedia_folder, options, self._strategy, h_ext, c_ext)

        # create main file with an example
        if ProjectFiles.MAIN in options.files:
            (text_example_h, text_main_c) = generate_embedia_main(embedia_model, self._src_lib_folder, self._dst_embedia_folder_name, model_name, options, self._strategy)
            
            # Get main filename and extension from strategy
            filename = self._strategy.get_main_filename(project_name)
            if options.project_type == ProjectType.ARDUINO:
                c_ext = '.ino'
            else:
                c_ext, h_ext = self._strategy.get_extensions()
            
            # Generate project file if strategy supports it
            project_content = self._strategy.generate_project_file(
                project_name, 
                self._get_project_files(embedia_model, options),
                self._src_lib_folder,
                self._dst_embedia_folder_name
            )
            if project_content:
                if options.project_type == ProjectType.CODEBLOCK:
                    project_filename = project_name + '.cbp'
                elif options.project_type in [ProjectType.CMAKE_C, ProjectType.CMAKE_CPP]:
                    project_filename = 'CMakeLists.txt'
                else:
                    project_filename = project_name + '.project'
                file_management.save_to_file(os.path.join(self._dst_folder, project_filename), project_content)

            file_management.save_to_file(os.path.join(self._dst_folder, filename + c_ext), text_main_c)
            if text_example_h is not None:
                file_management.save_to_file(os.path.join(self._dst_embedia_folder, 'example_file' + h_ext), text_example_h)

        if options.verbose:
            print(f'Project {project_name} exported in {os.path.abspath(output_folder)}')


    def _get_files_extension(self):
        return self._strategy.get_extensions()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
    def _datatype_subfolder(self, data_type):
        # absolute path for copying files
        if data_type == ModelDataType.FIXED8:
            dt_folder = 'fixed8'
        elif data_type == ModelDataType.FIXED16:
            dt_folder = 'fixed16'
        elif data_type == ModelDataType.FIXED32:
            dt_folder = 'fixed32'
        elif data_type == ModelDataType.QUANT8:
            dt_folder = 'quant8'
        elif data_type == ModelDataType.FULL_QUANT8:
            dt_folder = 'full_quant8'
        elif data_type == ModelDataType.BINARY:
            dt_folder = 'binary'
        elif data_type == ModelDataType.BINARY_FIXED32:
            dt_folder = 'binary&fixed32'
        elif data_type == ModelDataType.BINARY_FLOAT16:
            dt_folder = 'binary&float16'
        else:
            dt_folder = 'float'

        return f'mcu/{self._options.micro.lname}/{dt_folder}/'

    def _prepare_folders(self, output_folder, project_name, options):
        # create output folder if doesnt exists
        if output_folder[-1] != '/':
            output_folder += '/'

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder += project_name+'/'

        if options.clean_output:
            shutil.rmtree(path=output_folder, ignore_errors=True)


        self._dst_folder = os.path.abspath(output_folder)+'/'
        if not os.path.exists(self._dst_folder):
            os.mkdir(self._dst_folder)

        self._dst_embedia_folder = os.path.join(self._dst_folder, self._dst_embedia_folder_name)
        if not os.path.exists(self._dst_embedia_folder):
            os.mkdir(self._dst_embedia_folder)



    def _get_project_files(self, embedia_model, options):

        project_files = list()

        # main file and files extensions
        (c_ext, h_ext) = self._get_files_extension()
        project_files.append('main'+c_ext)

        # embedia files
        for (head_file, code_file) in embedia_model.required_files:
            if code_file:
                filename = code_file.name
                project_files.append(filename[0:-2] + c_ext)
            if head_file:
                filename = head_file.name
                project_files.append(filename[0:-2] + h_ext)

        # model files
        model_filename = embedia_model.model_name
        project_files.append(model_filename+c_ext)
        project_files.append(model_filename+h_ext)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
        hpp_ext = '.hpp'
        #half file
        if options.data_type == ModelDataType.BINARY_FLOAT16:
            project_files.append('half'+hpp_ext)
        elif options.data_type in [ModelDataType.QUANT8, ModelDataType.FULL_QUANT8]:
            project_files.append('quant8' + c_ext)
            project_files.append('quant8' + h_ext)
            if  options.data_type == ModelDataType.FULL_QUANT8:
                # fixed point files for support quantization operations
                project_files.append('fixed' + c_ext)
                project_files.append('fixed' + h_ext)
        elif options.data_type != ModelDataType.FLOAT and options.data_type != ModelDataType.BINARY and options.data_type != ModelDataType.BINARY_FLOAT16:
            # fixed point files
            project_files.append('fixed'+c_ext)
            project_files.append('fixed'+h_ext)

        # debug file
        if options.debug_mode != DebugMode.DISCARD:
            debug_fname = 'embedia_debug' if options.data_type != ModelDataType.FULL_QUANT8 else 'embedia_debug_quant'
            project_files.append(f'{debug_fname}'+c_ext)
            project_files.append(f'{debug_fname}'+h_ext)
            project_files.append('embedia_debug_def'+h_ext)

        # test examples
        if options.example_data is not None:
            project_files.append('example_file'+h_ext)

        return project_files

    def build_model_info(self, embedia_model):
        layers_info = embedia_model.get_layers_info()
        total_params = (0, 0)
        total_size = 0
        total_MACs = 0
        total_ACOPs = 0
        block_align = 4

        peak_bytes = 0
        peak_layer_name = ''
        peak_inp_sz = 0
        peak_out_sz = 0
        peak_int_sz = 0

        # --- data type info ---
        options = embedia_model.options
        is_quant8 = options.data_type == ModelDataType.QUANT8

        if is_quant8:
            compute_sz = 16 // 8
            compute_name = 'QUANT8 -> FIXED16'
            storage_sz = 1
            storage_name = 'QUANT8 (1 byte)'
        else:
            compute_sz = options.data_type.size // 8
            compute_name = f'{options.data_type.name} ({compute_sz} bytes)'
            storage_sz = compute_sz
            storage_name = f'{options.data_type.name} ({storage_sz} bytes)'

        storage_qualifier = options.model_storage.qualifier
        storage_dest = 'FLASH (const)' if storage_qualifier == 'const' else 'RAM'

        for i, (l_name, l_type, params, shape, MACs, ACOPs, size) in enumerate(layers_info):
            total_size += size
            total_MACs += MACs
            total_ACOPs += ACOPs
            total_params = (total_params[0] + params[0], total_params[1] + params[1])

            param_str = '%d' % (params[0] + params[1])
            if params[1] > 0:
                param_str += '(%d)' % params[1]

            # buffer: slot más grande — estadística de la capa
            buffer_sz = embedia_model.get_buffer_layer_size(i, block_align)
            buffer_str = '%8.3f' % (buffer_sz / 1024.0)

            # params: tamaño en storage (pesos + biases)
            size_str = '%8.3f' % (size / 1024.0)

            # peak: pico real de RAM en runtime (ambos slots simultáneos)
            working_sz = embedia_model._get_layer_working_size(i, block_align)
            if working_sz > peak_bytes:
                peak_bytes = working_sz
                peak_layer_name = l_name
                layer = embedia_model.embedia_layers[i]
                peak_inp_sz = layer.input_size * compute_sz
                peak_out_sz = layer.output_size * compute_sz
                peak_int_sz = layer.internal_alloc_required

            layers_info[i] = (l_type, l_name, param_str, shape,
                              MACs, ACOPs, buffer_str, size_str)

        alloc_buffer_sz = embedia_model.get_buffer_layer_max_size(block_align)

        # --- tabla ---
        table = PrettyTable()
        table.field_names = ['EmbedIA Layer', 'Name', '#Param(NT)', 'Shape',
                             'MACs', 'ACOPs', 'Buffer (KiB)', 'Params (KiB)']
        table.align['EmbedIA Layer'] = 'l'
        table.align['Name'] = 'l'
        table.align['#Param(NT)'] = 'r'
        table.align['MACs'] = 'r'
        table.align['ACOPs'] = 'r'
        table.align['Buffer (KiB)'] = 'r'
        table.align['Params (KiB)'] = 'r'

        for layer_info in layers_info:
            table.add_row(layer_info)

        model_info = '\n' + str(table) + '\n'

        # --- resumen ---
        total_p = '%d' % (total_params[0] + total_params[1])
        if total_params[1] > 0:
            total_p += '(%d)' % total_params[1]

        # peak breakdown
        if peak_layer_name:
            if peak_int_sz > 0:
                peak_detail = '  <- %s (inp=%d + out=%d + tmp=%d)' % (
                    peak_layer_name, peak_inp_sz, peak_out_sz, peak_int_sz)
            else:
                peak_detail = '  <- %s (inp=%d + out=%d)' % (
                    peak_layer_name, peak_inp_sz, peak_out_sz)
        else:
            peak_detail = ''

        model_info += 'Data types:\n'
        model_info += '  Compute : %s\n' % compute_name
        model_info += '  Storage : %s -> %s\n' % (storage_name, storage_dest)
        model_info += '\n'
        model_info += 'Total params (NT)....: %s\n' % total_p
        model_info += 'Total params (KiB)...: %.3f\n' % (total_size / 1024.0)
        model_info += 'Total MACs operations: %.0f\n' % total_MACs
        model_info += 'Total AC operations..: %.0f\n' % total_ACOPs
        model_info += 'Peak RAM (bytes).....: %d%s\n' % (alloc_buffer_sz, peak_detail)

        return model_info