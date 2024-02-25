import subprocess
from enum import Enum
import re


class CompilerResult(Enum):
    UNKNOWN = 0
    SUCCESS = 1
    WARNING = 2
    ERROR = 3
    def __str__(self):
        colors = {
            CompilerResult.UNKNOWN: '\033[0m',  # Reset color
            CompilerResult.SUCCESS: '\033[92m',  # Green
            CompilerResult.WARNING: '\033[93m',  # Yellow
            CompilerResult.ERROR: '\033[91m'  # Red
        }
        return f"{colors[self]}{self.name}{colors[CompilerResult.UNKNOWN]}"




def codeblocks_compile(project_name, codeblocks_path=None, target='Release'):
    '''
    Compiles a Code::Blocks project using command line.

    Args:
        project_name (str): the full path to the project file
            (e.g. 'C:/path_to_project/project_name.cbp').
        codeblocks_path (str, optional): the full path where the Code::Blocks
            executable is located. If not specified, it assumes that
            Code::Blocks is added to the PATH environment variable.
        target (str, optional): the type of compilation ('Debug' or 'Release').
            If not specified, it assumes that the project is compiled for
            Release.

    Returns:
        subprocess.CompletedProcess: an object that contains the result of the
        command execution.
    '''

    if codeblocks_path is None:
        # Assumes that Code::Blocks is added to the PATH environment variable
        codeblocks_path = 'codeblocks'
    else:
        if codeblocks_path[-1] != '/':
            codeblocks_path += '/'
        codeblocks_path = codeblocks_path + 'codeblocks'

    if target not in ['Debug', 'Release']:
        raise ValueError('Invalid target value. It should be "Debug" or "Release"')

    command = [codeblocks_path, '--build', project_name, '--target='+target]
    result = subprocess.run(command, capture_output=True, text=True)
    return (result, get_codeblocks_result(result))


def arduino_cli_compile(sketch_path, arduino_cli_path=None, board='arduino:avr:uno'):
    '''
    Compiles an Arduino sketch using command line.

    Args:
        sketch_path (str): the full path to the sketch file
            (e.g. 'C:/path_to_sketch/sketch_name.ino').
        arduino_path (str, optional): the full path where the Arduino
            executable is located. If not specified, it assumes that Arduino is
            added to the PATH environment variable.
        board (str, optional): the type of board to use for the compilation.
            If not specified, it assumes that the project is compiled for an
            'uno' board.

    Returns:
        subprocess.CompletedProcess: an object that contains the result of the
        command execution.
    '''

    if arduino_cli_path is None:
        # Assumes that arduino-cli vis added to the PATH environment variable
        arduino_cli_path = 'arduino-cli'
    else:
        if arduino_cli_path[-1] != '/':
            arduino_cli_path += '/'
        arduino_cli_path = arduino_cli_path + 'arduino-cli'

    command = [arduino_cli_path, 'compile', '-b', board, sketch_path]
    result = subprocess.run(command, capture_output=True, text=True)

    return (result, get_arduino_result(result))


def get_codeblocks_result(result):
    if result.stderr != '':
        return CompilerResult.ERROR
    match = re.search(r'(\d+)\s*(?=warning\()', result.stdout)
    if not match:
        return CompilerResult.UNKNOWN
    if int(match.group(1)):
        return CompilerResult.WARNING
    return CompilerResult.SUCCESS


def get_arduino_result(result):
    if result.stderr != '':
        return CompilerResult.ERROR

    return CompilerResult.SUCCESS


