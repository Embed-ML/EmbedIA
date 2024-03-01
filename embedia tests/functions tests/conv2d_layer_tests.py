from tensorflow.keras.layers import Conv2D

# Tests list
TESTS_LISTs = [
    # {'name': 'Fil2_Ker2x2_Str1x1_PadV_Inp4x4x1',  # Test filter, kernel fits dimensions
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(4, 4, 1))
    #  },
    # {'name': 'Fil2_Ker2x2_Str1x1_PadV_Inp5x4x1',  # Test filter, kernel fits W dimension only
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 4, 1))
    #  },
    #  {'name': 'Fil2_Ker2x2_Str1x1_PadV_Inp4x5x1',  # Test filter, kernel fits H dimension only
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(4, 5, 1))
    #  },
    #
    # {'name': 'Fil2_Ker3x3_Str1x1_PadV_Inp5x5x1',  # Test filter, kernel fits dimensions
    #  'element': Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 1))
    #  },
    # {'name': 'Fil2_Ker3x3_Str1x1_PadV_Inp5x4x1',  # Test filter, kernel fits W dimension only
    #  'element': Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 4, 1))
    #  },
    # {'name': 'Fil2_Ker3x3_Str1x1_PadV_Inp4x5x1',  # Test filter, kernel fits H dimension only
    #  'element': Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(4, 5, 1))
    #  },  # ---
    #
    # {'name': 'Fil2_Ker2x2_Str2x2_PadV_Inp5x5x1',  # Test filter, kernel doesn't fit dimensions
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 1))
    #  },
    # {'name': 'Fil2_Ker2x2_Str2x1_PadV_Inp5x5x1',  # Test asymmetrical stride w
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(2, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 1))
    #  },
    # {'name': 'Fil2_Ker2x2_Str1x2_PadV_Inp5x5x1',  # Test asymmetrical stride h
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 2), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 1))
    #  },
    # {'name': 'Fil2_Ker2x2_Str2x1_PadV_Inp5x5x3',  # Test asymmetrical stride w, multiple input channels
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(2, 1), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 3))
    #  },
    # {'name': 'Fil2_Ker2x2_Str1x2_PadV_Inp5x5x3',  # Test asymmetrical stride h, multiple input channels
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 2), padding='valid', activation='linear',
    #                    input_shape=(5, 5, 3))
    #  },
    # {'name': 'Fil2_Ker2x2_Str1x2_PadS_Inp5x5x3',  # Test asymmetrical stride h, multiple input channels
    #  'element': Conv2D(filters=2, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='linear',
    #                    input_shape=(5, 5, 3))
    #  },

]

# Función para inicializar la lista
def init_tests_lists():

    tests = []
    for filters in [1, 2]:
        for kernel_size in [(2,2), (3,3)]:
            for strides in [(1,1), (1,2), (2,1), (2,2)]:
                for padding in ['valid', 'same']:
                    for input_shape in [(6, 6, 1), (5, 5, 1), (5, 5, 3)]:
                        name = f'Fil{filters}_Ker{kernel_size}_Str{strides}_Pad{padding[0].upper()}_Inp{input_shape}'
                        test = {'name': name,
                                'element': Conv2D(filters=filters,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               activation='linear',
                                               input_shape=input_shape)
                             }
                        tests.append(test)
    return tests

TESTS_LIST = init_tests_lists()

if __name__ == '__main__':
    import sys

    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary


    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))
