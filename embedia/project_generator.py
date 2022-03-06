# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 00:33:13 2021

@author: cesar
"""
import os
import shutil

from embedia.model_generator.models import create_model_template_c, create_main_template_c, format_model_name
from embedia.model_generator.utils.generator_utils import create_codeblock_project
from embedia.project_options import ModelDataType, ProjectType, ProjectFiles, DebugMode
    


class ProjectGenerator:
    _dst_folder = None
    _root_folder = 'libraries/'
    _src_files_folder = None
    _src_dbg_folder = None
    
    def create_project(self, output_folder, project_name, model, options):
        
        self._prepare_folders(output_folder, project_name, options)

        project_files = list()
        
        # extension of files to copy/create
        if options.project_type in [ProjectType.C, ProjectType.CODEBLOCK]:
            (h_ext, c_ext)  = ('.h', '.c')
        else:
            (h_ext, c_ext)  = ('.h', '.cpp')
                    
            
        # create model files
        if ProjectFiles.MODEL in options.files:
            (c_code, c_header, filename) = create_model_template_c(model, options)
           
            self._save_to_file(self._dst_folder+filename+h_ext, c_header)
            self._save_to_file(self._dst_folder+filename+c_ext, c_code)
            

        
        # copy library files       
        if ProjectFiles.LIBRARY in options.files:

            
            if options.project_type == ProjectType.ARDUINO:
                shutil.copy(os.path.join(self._src_files_folder,'embedia.h'), os.path.join(self._dst_folder,'embedia'+h_ext))
                content = self._read_from_file(os.path.join(self._src_files_folder,'embedia.c'))
                #add include
                content.insert(7,'#include "Arduino.h"\n')
                
                self._save_to_file(os.path.join(self._dst_folder,'embedia'+c_ext),''.join(content))
            else:
                content = self._read_from_file(os.path.join(self._src_files_folder,'embedia.h'))
                #add include
                content.insert(7, '#include <stdlib.h>\n')
                self._save_to_file(os.path.join(self._dst_folder,'embedia'+h_ext),''.join(content))

                shutil.copy(os.path.join(self._src_files_folder,'embedia.c'), os.path.join(self._dst_folder,'embedia'+c_ext))
            
            if options.data_type != ModelDataType.FLOAT:
                
                shutil.copy(os.path.join(self._src_files_folder,'fixed.c'), os.path.join(self._dst_folder,'fixed'+c_ext))
                shutil.copy(os.path.join(self._src_files_folder,'fixed.h'), os.path.join(self._dst_folder,'fixed'+h_ext))

           

        # copy debug files       
        if options.debug_mode != DebugMode.DISCARD:
            
            # add debug mode macro to header file
            content = self._read_from_file(os.path.join(self._src_dbg_folder,'embedia_debug.h'))
            #add include
            content.insert(7,'#define EMBEDIA_DEBUG %d\n' % options.debug_mode)
            self._save_to_file(os.path.join(self._dst_folder, 'embedia_debug.h'),''.join(content))

            # copy aditional debug file
            if options.project_type == ProjectType.ARDUINO:
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug_def_arduino.h'), os.path.join(self._dst_folder, 'embedia_debug_def.h'))
                # copy implementation file
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug.c'), os.path.join(self._dst_folder, 'embedia_debug.cpp'))
            else:
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug_def_c.h'), os.path.join(self._dst_folder, 'embedia_debug_def.h'))
                # copy implementation file
                shutil.copy(os.path.join(self._src_dbg_folder, 'embedia_debug.c'), os.path.join(self._dst_folder, 'embedia_debug.c'))



        # create main file with an example
        if ProjectFiles.MAIN in options.files:
            c_main = create_main_template_c(model=model, 
                                            example=options.example_data, 
                                            example_comment=options.example_comment, options=options);
            # change project extension for arduino
            if options.project_type == ProjectType.ARDUINO:
                main_name = project_name+'.ino'
            else: 
                main_name = 'main.c'
                if options.project_type == ProjectType.CODEBLOCK:
                    model_name = format_model_name(model)
                    files = self._get_project_files(model_name, options)
                    project = create_codeblock_project(project_name, files)
                    self._save_to_file(os.path.join(self._dst_folder, project_name+'.cbp'), project)
                    
                    
            self._save_to_file(self._dst_folder+main_name, c_main)
                               
          
    def _prepare_folders(self, output_folder, project_name, options):
        # create output folder if doesnt exists
        if output_folder[-1] != '/':
            output_folder+='/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        
        
        # Arduino require a project folder with same name as main file
        #if options.project_type == ProjectType.ARDUINO:
        output_folder+=project_name+'/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        
        # absolute path for copying files
        if options.data_type == ModelDataType.FLOAT:
            src_folder = 'float'
        elif options.data_type == ModelDataType.FIXED8:
            src_folder = 'fixed8'
        elif options.data_type == ModelDataType.FIXED16:
            src_folder = 'fixed16'
        elif options.data_type == ModelDataType.FIXED32:
            src_folder = 'fixed32'
        
        self._src_files_folder = os.path.abspath(self._root_folder+ src_folder)+'/'
        self._dst_folder = os.path.abspath(output_folder)+'/'
        self._src_dbg_folder = os.path.abspath(self._root_folder+'debug')+'/'
        
        
    def _get_project_files(self, model_filename, options):
        
        project_files = list()
        
        # main file and files extensions
        if options.project_type in [ProjectType.C, ProjectType.CODEBLOCK]:
            (h_ext, c_ext)  = ('.h', '.c')
            project_files.append('main.c')
        else:
            (h_ext, c_ext)  = ('.h', '.cpp') 
            if options.project_type != ProjectType.ARDUINO:
                project_files.append('main.cpp')
            
        # embedia files
        project_files.append('embedia'+c_ext)
        project_files.append('embedia'+h_ext)

        # model files
        project_files.append(model_filename+c_ext)
        project_files.append(model_filename+h_ext)
        
        # fixed point files
        if options.data_type != ModelDataType.FLOAT:
            project_files.append('fixed'+c_ext)
            project_files.append('fixed'+h_ext)
                    
        # debug files       
        if options.debug_mode != DebugMode.DISCARD:
            project_files.append('embedia_debug'+c_ext)
            project_files.append('embedia_debug'+h_ext)
            project_files.append('embedia_debug_def'+h_ext)
            
        return project_files

            
    def _save_to_file(self,filename, content):
        file = open(filename, 'w', encoding='utf-8') #guardar
        file.write(content)
        file.close()   
        
    def _read_from_file(self,filename):
        file = open(filename, 'r', encoding='utf-8') #guardar
        content = file.readlines()
        file.close()  
        return content
        
    
        
    
    
    