from embedia.model_generator.project_options import ModelDataType

class TestsConfig:

    EMBEDIA_PATH = '../../embedia/'
    OUTPUT_PATH = './outputs/'
    KEEP_SUCCESS_TESTS = False

    DIM_1D_INPUT = 50
    DIM_1D_OUTPUT = 50
    DIM_2D_INPUT = (50, 50)
    DIM_2D_OUTPUT = (50, 50)
    DIM_3D_INPUT = (50, 50, 3)
    DIM_3D_OUTPUT = (50, 50, 3)

    @property
    def DATA_TYPES(self):
        return [ModelDataType.FLOAT, ModelDataType.FIXED32, ModelDataType.FIXED16, ModelDataType.FIXED8, ModelDataType.QUANT8]
        # return [ModelDataType.FLOAT, ModelDataType.QUANT8]
        # return [ModelDataType.QUANT8]
        # return [ModelDataType.FLOAT]

    @property
    def DATA_TYPE_BOUND_ERROR(self):
        # order of DATA_TYPES property
        return [1e-5, 5e-4, 0.025, 0.5, 0.05]
        # return [1e-1, 5e-1, 0.2, 1, 0.01]

