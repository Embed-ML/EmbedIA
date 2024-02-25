import platform
import os
from enum import Enum
import subprocess


class OperatingSystem(Enum):
    WINDOWS = 1
    LINUX = 2
    MACOS = 3
    UNKNOWN = 4


class SystemPlatformProvider:

    def __init__(self):
        self._os = self._detect_operating_system()

    @property
    def os(self):
        return self._os

    def _detect_operating_system(self) -> OperatingSystem:
        """
        Detects the operating system and returns an enumerated value representing the OS.

        Returns:
            OperatingSystem: An enumerated value representing the detected operating system.
        """
        os_name = platform.system().lower()
        if 'windows' in os_name:
            return OperatingSystem.WINDOWS
        elif 'linux' in os_name:
            return OperatingSystem.LINUX
        elif 'darwin' in os_name:  # macOS is based on Darwin
            return OperatingSystem.MACOS
        else:
            return OperatingSystem.UNKNOWN

    def _run_command(self, command) -> tuple:
        """
        Executes a command in the command line and returns the result as a string.

        Args:
            command (str): The command to execute.

        Returns:
            tuple: A tuple containing the return code and the output (stdout or stderr) of the command execution.
        """
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return (result.returncode, result.stdout)
            else:
                return (result.returncode, result.stderr)
        except Exception as e:
            print(f"Error executing command: {e}")
            return (None, '')

    def _format_command(self, command: str, params: list) -> str:
        # Verificar que los parámetros no estén vacíos
        if not params:
            return command

        formatted_command = command
        # Formatear los parámetros como una cadena separada por espacios
        for param in params:
            if param[0] !='"':
                param = '"'+param+'"'
            formatted_command+=' '+param

        return formatted_command

    def run(self, command: str, params: list =[]) -> tuple:
        """
        Executes a command with the given parameters and returns the result as a tuple.

        Args:
            command (str): The command to execute.
            params (list): A list containing the parameters for the command.

        Returns:
            tuple: A tuple containing the return code, stdout/stderr of the command execution.
        """

        formatted_command = self._format_command(command, params)
        result = subprocess.run(formatted_command, capture_output=True, text=True, shell=True)

        if result.returncode == 0:
            return (result.returncode, result.stdout)
        return (result.returncode, result.stderr)


class GccCompiler(SystemPlatformProvider):
    def __init__(self):
        super().__init__()
        self._gcc_path = self._find_gcc_path()

    @property
    def gcc_path(self):
        return self._gcc_path

    def _find_gcc_path(self)->str:
        """
        Finds the installation path of GCC on the system.

        Returns:
            str: The installation path of GCC if found, or an empty string if not found.
        """
        os_type = self.os

        if os_type == OperatingSystem.WINDOWS:
            # Check if GCC is installed in the default location on Windows
            # TO DO: implement autodetection
            for mingw_folder in ["MinGW64", "MinGW32", "MinGW"]:
                default_path = os.path.join("C:", os.sep, mingw_folder, "bin", "gcc.exe")
                if os.path.exists(default_path):
                    return default_path
        elif os_type in (OperatingSystem.LINUX, OperatingSystem.MACOS):
            # Use the 'which' command to find the GCC path on Linux and macOS
            try:
                result = run(['which', 'gcc'], stdout=PIPE, stderr=PIPE, universal_newlines=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception as e:
                print(f"Error finding GCC path: {e}")

        return ""

    def compile(self, output_path, main, files:list)->None:
        params = []
        params.extend(['-o', os.path.join(output_path, main.replace('.c', '') + '.exe')])
        params.extend(files)
        return self.run(self.gcc_path, params)


if __name__ == "__main__":
    compiler = GccCompiler()
    print(f"Operating System: {compiler.os}")
    print(f"GCC Path: {compiler.gcc_path}")