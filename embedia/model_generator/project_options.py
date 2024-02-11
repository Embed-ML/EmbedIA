class ModelDataType:
    (FLOAT, FIXED32, FIXED16, FIXED8, QUANT8, BINARY, BINARY_FIXED32, BINARY_FLOAT16) = (0, 1, 2, 3, 4, 5, 6, 7)

    SIZES = [32, 32, 16, 8, 8, 32, 32, 16]

    def get_size(dt):

        return ModelDataType.SIZES[dt]


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
        self.embedia_folder = None
        self.project_type = ProjectType.C
        self.data_type = ModelDataType.FLOAT
        self.baud_rate = 9600
        self.example_data = None
        self.example_labels = None
        self.files = ProjectFiles.ALL
        self.debug_mode = DebugMode.DISABLED
        self.clean_output = False
        self.normalizer = None
        self.tamano_bloque = BinaryBlockSize.Bits8
        self.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_KNOWN