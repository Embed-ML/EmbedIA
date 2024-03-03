from tensorflow.keras.layers import SeparableConv2D

# Tests list
# TESTS_LIST = [
#     {'name': 'Ker3x3_Str1x1_Pad0_Mul1_Inp6x6x1', # simple test
#      'element': SeparableConv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
#                         depth_multiplier=1, input_shape=(6,6,1))
#      }
#     ]

def init_tests_lists():
    tests = []
    for filters in [1,2]:
        for kernel_size in [(2,2), (2,3), (3,2), (3,3)]:
            for strides in [(1,1)]:
                for padding in ['valid']:
                    for input_shape in [(6, 6, 1), (5, 5, 3)]:
                        name = f'Fil{filters}_Ker{kernel_size}_Str{strides}_Pad{padding[0].upper()}_Inp{input_shape}'
                        test = {'name': name,
                                'element': SeparableConv2D(filters=filters,
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
