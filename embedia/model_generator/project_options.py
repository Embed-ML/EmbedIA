from enum import Enum, IntEnum, IntFlag

from IPython.utils.terminal import set_term_title

MODEL_DATA_TYPE_SIZES = (32, 32, 16, 8, 8, 8, 32, 32, 16)  # Bit sizes for each data type

class ModelDataType(Enum):
    """
    Enumeration of supported data types for model quantization and inference.
    Each type has an associated bit size and string name.
    """
    FLOAT             = 0
    FIXED32           = 1
    FIXED16           = 2
    FIXED8            = 3
    QUANT8            = 4
    FULL_QUANT8       = 5
    BINARY            = 6
    BINARY_FIXED32    = 7
    BINARY_FLOAT16    = 8


    @property
    def size(self):
        """Returns the bit size of the data type."""
        return MODEL_DATA_TYPE_SIZES[self.value]

    @property
    def lname(self):
        """Returns the lowercase name of the data type."""
        return self.name.lower()

    @property
    def is_fixed_point(self):
        """Returns True if the data type is a fixed-point representation."""
        return self in {ModelDataType.FIXED8, ModelDataType.FIXED16, ModelDataType.FIXED32}

    @property
    def is_quantized(self):
        """Returns True if the data type is a quantized representation."""
        return self in {ModelDataType.QUANT8, ModelDataType.FULL_QUANT8}

    @property
    def is_binary(self):
        """Returns True if the data type is a binary representation."""
        return self in {ModelDataType.BINARY, ModelDataType.BINARY_FIXED32, ModelDataType.BINARY_FLOAT16}


class StorageType(Enum):
    """Estrategia de almacenamiento para parámetros del modelo"""
    VOLATILE = 1   # RAM: en memoria volátil, sin modificadores
    PERSISTENT = 2 # FLASH: en memoria persistente, con modificador 'const'

    @property
    def qualifier(self) -> str:
        """Deriva el qualifier de C necesario"""
        return 'const' if self == StorageType.PERSISTENT else ''


class ModelMicro(Enum):
    """
    Target microcontroller platforms for hardware-specific optimizations.
    """
    GENERIC = 0
    ESP32   = 1

    @property
    def lname(self):
        """Returns the string name of the microcontroller."""
        return self.name.lower()


class ProjectType(Enum):
    """
    Type of project to generate (affects file structure and code style).
    """
    C         = 0
    CPP       = 1
    ARDUINO   = 2
    CODEBLOCK = 3  # For Code::Blocks IDE projects
    CMAKE_C   = 4  # CMake-based C project
    CMAKE_CPP = 5  # CMake-based C++ project


class ProjectFiles(IntFlag):
    """
    Files to include in the exported project.
    Uses IntEnum to support bitmask operations if needed.
    """
    LIBRARY = 1  # embedia library files
    MAIN    = 2  # main application file (e.g., main.c)
    MODEL   = 4  # model file (e.g., model_data.c)

    ALL = LIBRARY | MAIN | MODEL



class DebugMode(IntEnum):
    """
    Level of debug information to include in the generated code.
    Negative values are allowed (e.g., -1 for special behavior).
    """
    DISCARD   = -1  # Discard all debug info
    DISABLED  = 0   # No debug output
    HEADERS   = 1   # Include debug headers
    DATA      = 2   # Include full data dumps


class BinaryBlockSize(Enum):
    """
    Block size (in bits) for packing binary weights in binary neural networks.
    """
    Bits8  = 0
    Bits16 = 1
    Bits32 = 2
    Bits64 = 3


class UnimplementedLayerAction(Enum):
    """
    Action to take when an unimplemented layer is found during model export.
    """
    FAILURE        = 0  # Raise an error and stop
    IGNORE_ALL     = 1  # Skip the layer silently
    IGNORE_KNOWN   = 2  # Skip only known unimplemented layers (with warning)


class ProjectOptions:
    """
    Configuration options for project generation.
    This class holds all user-defined settings for the export process.
    """
    def __init__(self):
        self.embedia_folder = None            # embedia source folder
        self.project_type = ProjectType.C     # project type to export
        self.micro = ModelMicro.GENERIC       # microcontroller for hardware optimization
        self._data_type = ModelDataType.FLOAT # data type for data storage
        self.baud_rate = 9600                 # Arduino Only. Set Serial device speed
        self.example_data = None              # list of examples to include in project
        self.example_labels = None            # list of labels for examples (classification)
        self.files = ProjectFiles.ALL         # set of files to export library, main or model
        self.debug_mode = DebugMode.DISABLED  # debug info to include and what to show
        self.clean_output = False             # clear output folder before export (use carefully)
        self.preprocessing = None             # preprocessing objects to add before start inference
        self.postprocessing = None            # postprocessing objects to add after last layer
        self.tamano_bloque = BinaryBlockSize.Bits8 # block size for binary nets
        self.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_KNOWN # error action when find an unimplemented layer
        self.output_subfolder = 'embedia'     # name of folder to store all embedia files
        self.verbose = True                   # verbose output during project generation
        self._fixed_precision = None          # number of fractional bits for fixed-point data types (None for default)
        self.model_storage = StorageType.PERSISTENT # storage type for model parameters (e.g., weights, data structures, etc.)
    
    @property
    def fixed_precision(self):
        """Number of fractional bits for fixed-point data types."""
        return self._fixed_precision
    
    @fixed_precision.setter
    def fixed_precision(self, value:int):
        """Sets the number of fractional bits for fixed-point data types."""
        if self.data_type.is_fixed_point:
            if value < 0 or value > self.data_type.size - 1:
                raise ValueError(f'Fixed precision must be between 0 and {self.data_type.size - 1} for {self.data_type.name}')
        self._fixed_precision = value

    @property
    def data_type(self):
        """Data type for model parameters and computations."""
        return self._data_type

    @data_type.setter
    def data_type(self, value:ModelDataType):
        """Sets the data type. Set data and set fixed_precision to half of data type size if it is out of range."""

        self._data_type = value
        # Validate fixed precision if current data type is fixed-point
        if value.is_fixed_point:
            if self._fixed_precision is None or self._fixed_precision < 0 or self._fixed_precision > self.data_type.size - 1:
                self._fixed_precision = self.data_type.size // 2

