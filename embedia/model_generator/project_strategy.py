"""
Project generation strategies for different project types.
Implements the Strategy pattern to handle project-specific generation logic.
"""
import os
from embedia.utils.c_helper import CBuilder, ArduinoBuilder
from embedia.utils import file_management
from embedia.model_generator.project_options import ProjectType


class ProjectStrategy:
    """Base strategy - defines the interface for all project generators."""
    
    def get_extensions(self):
        """Returns (code_ext, header_ext) tuple."""
        raise NotImplementedError
    
    def get_coder(self):
        """Returns the code builder instance."""
        raise NotImplementedError

    def get_includes(self):
        """Returns include statements for main file."""
        raise NotImplementedError

    def get_baud_rate(self, options):
        """Returns baud rate string (Arduino only)."""
        return "\n"

    def get_embedia_folder_name(self, options):
        """Returns the subfolder name for embedia files."""
        return options.output_subfolder

    def get_main_filename(self, project_name):
        """Returns the main file name (without extension)."""
        return "main"

    def get_debug_files(self, debug_filename):
        """Returns (def_header, impl_file) tuple."""
        return ('embedia_debug_def_c.h', f'{debug_filename}.c')

    def generate_project_file(self, project_name, files, src_folder, dst_folder):
        """Generates project-specific file (e.g., .cbp, CMakeLists.txt). Returns content or None."""
        return None


class CProjectStrategy(ProjectStrategy):
    """Strategy for plain C projects."""
    
    def get_extensions(self):
        return ('.c', '.h')
    
    def get_coder(self):
        return CBuilder()
    
    def get_includes(self):
        return '#include <stdio.h>\n'


class CPPProjectStrategy(ProjectStrategy):
    """Strategy for plain C++ projects."""
    
    def get_extensions(self):
        return ('.cpp', '.h')
    
    def get_coder(self):
        return CBuilder()
    
    def get_includes(self):
        return '#include <cstdio>\n'


class ArduinoProjectStrategy(ProjectStrategy):
    """Strategy for Arduino projects."""
    
    def get_extensions(self):
        return ('.cpp', '.h')
    
    def get_coder(self):
        return ArduinoBuilder()
    
    def get_includes(self):
        return '#include "Arduino.h"\n'
    
    def get_baud_rate(self, options):
        return str(options.baud_rate)
    
    def get_embedia_folder_name(self, options):
        return ''
    
    def get_main_filename(self, project_name):
        return project_name
    
    def get_debug_files(self, debug_filename):
        return ('embedia_debug_def_arduino.h', f'{debug_filename}.cpp')


class CodeBlockProjectStrategy(CProjectStrategy):
    """Strategy for Code::Blocks IDE projects."""
    
    def generate_project_file(self, project_name, files, src_folder, dst_folder):
        """Generates .cbp project file for Code::Blocks."""
        embedia_output_folder = dst_folder
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
        content = content.replace('{project_name}', project_name)
        content = content.replace('{included_files}', included_files)
        
        return content


class CMakeCProjectStrategy(CProjectStrategy):
    """Strategy for CMake-based C projects."""
    
    def generate_project_file(self, project_name, files, src_folder, dst_folder):
        """Generates CMakeLists.txt for C project."""
        c_files = [f for f in files if f.endswith('.c')]
        h_files = [f for f in files if f.endswith('.h')]
        
        # Separate main.c from embedia files
        main_file = 'main.c' if 'main.c' in c_files else ''
        embedia_files = [f for f in c_files if f != 'main.c']
        
        # Build source list
        sources = []
        if main_file:
            sources.append(main_file)
        sources.extend([os.path.join(dst_folder, f) for f in embedia_files])
        
        sources_str = '\n    '.join(sources)
        
        cmake_content = f'''cmake_minimum_required(VERSION 3.10)
project({project_name} C)

set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(${{CMAKE_SOURCE_DIR}}/{dst_folder})

# Source files
add_executable({project_name}
    {sources_str}
)
'''
        return cmake_content


class CMakeCPPProjectStrategy(CPPProjectStrategy):
    """Strategy for CMake-based C++ projects."""
    
    def generate_project_file(self, project_name, files, src_folder, dst_folder):
        """Generates CMakeLists.txt for C++ project."""
        cpp_files = [f for f in files if f.endswith('.cpp')]
        h_files = [f for f in files if f.endswith('.h') or f.endswith('.hpp')]
        
        # Separate main file from embedia files
        main_file = None
        for f in cpp_files:
            if 'main' in f or f.endswith('.ino'):
                main_file = f
                break
        
        embedia_files = [f for f in cpp_files if f != main_file]
        
        # Build source list
        sources = []
        if main_file:
            sources.append(main_file)
        sources.extend([os.path.join(dst_folder, f) for f in embedia_files])
        
        sources_str = '\n    '.join(sources)
        
        cmake_content = f'''cmake_minimum_required(VERSION 3.10)
project({project_name} CXX)

set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(${{CMAKE_SOURCE_DIR}}/{dst_folder})

# Source files
add_executable({project_name}
    {sources_str}
)
'''
        return cmake_content


# Strategy registry
PROJECT_STRATEGIES = {
    ProjectType.C:         CProjectStrategy(),
    ProjectType.CPP:       CPPProjectStrategy(),
    ProjectType.ARDUINO:   ArduinoProjectStrategy(),
    ProjectType.CODEBLOCK: CodeBlockProjectStrategy(),
    ProjectType.CMAKE_C:   CMakeCProjectStrategy(),
    ProjectType.CMAKE_CPP: CMakeCPPProjectStrategy(),
}


def get_project_strategy(project_type):
    """Factory function to get the appropriate strategy for a project type."""
    strategy = PROJECT_STRATEGIES.get(project_type)
    if strategy is None:
        raise ValueError(f"Unsupported project type: {project_type}")
    return strategy
