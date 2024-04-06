class ModelDataType:
    (FLOAT, FIXED32, FIXED16, FIXED8, QUANT8, BINARY, BINARY_FIXED32, BINARY_FLOAT16) = (0, 1, 2, 3, 4, 5, 6, 7)

    SIZES = [32, 32, 16, 8, 8, 32, 32, 16]
    NAMES = ['float', 'fixed32', 'fixed16', 'fixed8', 'quant8', 'binary', 'binary_fixed32', 'binary_fixed16']
    def get_size(dt):
        return ModelDataType.SIZES[dt]

    def get_name(dt):
        return ModelDataType.NAMES[dt]


class ProjectType:
    (C, CPP, ARDUINO, CODEBLOCK) = (0, 1, 2, 3)


class ProjectFiles:
    (LIBRARY, MAIN, MODEL) = (1, 2, 4)
    (ALL) = {LIBRARY, MAIN, MODEL}


class DebugMode:
    (DISCARD, DISABLED, HEADERS, DATA) = (-1, 0, 1, 2)


class BinaryBlockSize:
    (Bits8, Bits16, Bits32, Bits64) = (0, 1, 2, 3)


class UnimplementedLayerAction:
    (FAILURE, IGNORE_ALL, IGNORE_KNOWN) = (0, 1, 2)


class ProjectOptions:
    def __init__(self):
        self.embedia_folder = None           # embedia source folder
        self.embedia_output_subfloder = ''   # subfolder in output folder to copy embedia files
        self.project_type = ProjectType.C    # project type to export
        self.data_type = ModelDataType.FLOAT # data type for data storage
        self.baud_rate = 9600                # Arduino Only. Set Serial device speed
        self.example_data = None             # list of examples to include in project
        self.example_labels = None           # list of labels for examples (classification)
        self.files = ProjectFiles.ALL        # set of files to export library, main or model
        self.debug_mode = DebugMode.DISABLED # debug info to include and what to show
        self.clean_output = False            # clear output folder before export (use carefully)
        self.normalizer = None               # normalization object to add before start inference
        self.tamano_bloque = BinaryBlockSize.Bits8 # block size for binary nets
        self.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_KNOWN # error action when find an unimplemented layer
        self.output_subfolder = 'embedia'    # name of folder to store all embedia files