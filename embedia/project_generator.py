import os
import shutil

from embedia.core.model_factory import ModelFactory
from embedia.model_generator.project_options import (
        ModelDataType,
        ProjectType,
        ProjectFiles,
        DebugMode
    )

from embedia.model_generator.generate_files import (
        generate_embedia_library,
        generate_embedia_model,
        generate_embedia_main,
        generate_embedia_debug,
        generate_codeblock_project
    )

from embedia.utils import file_management

from prettytable import PrettyTable

def format_model_name(model):
    model_name = model.name.lower()
    if not model_name.endswith('model'):
        model_name += '_model'

    return model_name


class ProjectGenerator:

    def __init__(self, options):
        self._dst_folder = None
        self._options = options

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
        self._lib_folder = self._root_folder + 'libraries/'
        self._datatype_folder = self._lib_folder + self._datatype_subfolder(self._options.data_type)
        self._src_dbg_folder = self._lib_folder + 'debug/'

    def create_project(self, output_folder, project_name, model, options):

        embedia_model = ModelFactory.create_model(model, options)

        embedia_layers = embedia_model.embedia_layers

        # prepare folders and extension of files to copy/create
        self._prepare_folders(output_folder, project_name, options)

        c_ext, h_ext = self._get_files_extension()

        # copy library files
        if ProjectFiles.LIBRARY in options.files:
            embedia_files = generate_embedia_library(embedia_model, self._datatype_folder, self._dst_folder, h_ext, c_ext, options)

            # print layers memory size
            model_info = self.build_model_info(embedia_model, self._extract_datatypes(embedia_files))
            print(model_info)

        # create model files
        if ProjectFiles.MODEL in options.files:
            (text_model_h, text_model_c, filename) = generate_embedia_model(embedia_model, self._lib_folder, self._dst_folder, h_ext, c_ext, model.name, model_info, options)

        # copy debug file
        if options.debug_mode != DebugMode.DISCARD:
            generate_embedia_debug(self._src_dbg_folder, self._dst_folder, options)

        # create main file with an example
        if ProjectFiles.MAIN in options.files:
            (text_example_h, text_main_c) = generate_embedia_main(embedia_layers, self._lib_folder, filename, options, embedia_model)
            if options.project_type == ProjectType.ARDUINO:
                filename = project_name
                c_ext = '.ino'
            else:
                filename = "main"
                if options.project_type == ProjectType.CODEBLOCK:
                    model_name = format_model_name(model)
                    files = self._get_project_files(model_name, options)
                    project = generate_codeblock_project(project_name, files, self._lib_folder)
                    file_management.save_to_file(os.path.join(self._dst_folder, project_name+'.cbp'), project)

            file_management.save_to_file(os.path.join(self._dst_folder, filename + c_ext), text_main_c)
            if text_example_h is not None:
                file_management.save_to_file(os.path.join(self._dst_folder, 'example_file' + h_ext), text_example_h)

    def _get_files_extension(self):
        if self._options.project_type in [ProjectType.C, ProjectType.CODEBLOCK]:
            (c_ext, h_ext) = ('.c', '.h')
        else:
            (c_ext, h_ext) = ('.cpp', '.h')
        return c_ext, h_ext

    def _extract_datatypes(self, embedia_files):
        (c_ext, h_ext) = self._get_files_extension()
        embedia_headers = ''
        for filename in embedia_files:
            if filename.endswith(h_ext):
                with open(os.path.join(self._datatype_folder, filename), 'r') as file:
                    embedia_headers += file.read()
        return embedia_headers

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
    def _datatype_subfolder(self, data_type):
        # absolute path for copying files
        if data_type == ModelDataType.FIXED8:
            return 'fixed8/'
        elif data_type == ModelDataType.FIXED16:
            return 'fixed16/'
        elif data_type == ModelDataType.FIXED32:
            return 'fixed32/'
        elif data_type == ModelDataType.QUANT8:
            return 'quant8/'
        elif data_type == ModelDataType.BINARY:
            return 'binary/'
        elif data_type == ModelDataType.BINARY_FIXED32:
            return 'binary&fixed32/'
        elif data_type == ModelDataType.BINARY_FLOAT16:
            return 'binary&float16/'
        return 'float/'

    def _prepare_folders(self, output_folder, project_name, options):
        # create output folder if doesnt exists
        if output_folder[-1] != '/':
            output_folder += '/'

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder += project_name+'/'

        if options.clean_output:
            shutil.rmtree(path=output_folder, ignore_errors=True)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self._dst_folder = os.path.abspath(output_folder)+'/'

    def _get_project_files(self, model_filename, options):

        project_files = list()
        hpp_ext = '.hpp'
        # main file and files extensions
        (c_ext, h_ext) = self._get_files_extension()
        project_files.append('main'+c_ext)

        # embedia files
        project_files.append('embedia'+c_ext)
        project_files.append('embedia'+h_ext)

        # model files
        project_files.append(model_filename+c_ext)
        project_files.append(model_filename+h_ext)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! editado
        #half file
        if options.data_type == ModelDataType.BINARY_FLOAT16:
            project_files.append('half'+hpp_ext)
        elif options.data_type == ModelDataType.QUANT8:
            project_files.append('quant8' + c_ext)
            project_files.append('quant8' + h_ext)
        elif options.data_type != ModelDataType.FLOAT and options.data_type != ModelDataType.BINARY and options.data_type != ModelDataType.BINARY_FLOAT16:
            # fixed point files
            project_files.append('fixed'+c_ext)
            project_files.append('fixed'+h_ext)

        # debug file
        if options.debug_mode != DebugMode.DISCARD:
            project_files.append('embedia_debug'+c_ext)
            project_files.append('embedia_debug'+h_ext)
            project_files.append('embedia_debug_def'+h_ext)

        # test examples
        if options.example_data is not None:
            project_files.append('example_file'+h_ext)

        return project_files

    def build_model_info(self, embedia_model, embedia_headers):
        embedia_decl = embedia_headers
        layers_info = embedia_model.get_layers_info(embedia_decl)
        total_params = (0, 0)
        total_size = 0
        total_MACs = 0

        for i, (l_name, l_type, params, shape, MACs, size) in enumerate(layers_info):
            total_size += size
            total_MACs += MACs
            total_params = (total_params[0] + params[0], total_params[1] + params[1])
            size = '%8.3f' % (size/1024.0)
            param_str= '%d' % (params[0] + params[1])
            if params[1] > 0:
                param_str += '(%d)' % params[1]
            layer = l_type
            layers_info[i] = (layer, l_name, param_str, shape, MACs, size)
        # print table
        table = PrettyTable()
        table.field_names = ['EmbedIA Layer', 'Name', '#Param(NT)', 'Shape', 'MACs', 'Size (KiB)']
        table.align['EmbedIA Layer'] = 'l'
        table.align['Name'] = 'l'
        table.align['#Param(NT)'] = 'r'
        table.align['MACs'] = 'r'
        table.align['Size (KB)'] = 'r'
        table.align['#Param'] = 'r'
#         table.add_rows(layers_info)
        for layer_info in layers_info:
            table.add_row(layer_info)
        model_info = '\n'+str(table)+'\n'

        total_p = '%d' % (total_params[0] + total_params[1])
        if total_params[1] > 0:
            total_p += '(%d)' % total_params[1]
        model_info += 'Total params (NT)....: %s\n' % total_p
        model_info += 'Total size in KiB....: %.3f\n' % (total_size/1024.0)
        model_info += 'Total MACs operations: %.0f\n' % total_MACs

        return model_info
