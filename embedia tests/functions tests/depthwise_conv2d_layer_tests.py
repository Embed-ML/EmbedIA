from tensorflow.keras.layers import DepthwiseConv2D

# Tests list
# TESTS_LIST = [
#     {'name': '_Inspect', # simple test
#      'element': DepthwiseConv2D(kernel_size=(2, 3), strides=(1, 1), padding='same', activation='linear',
#                         bias_initializer='random_normal', input_shape=(6,6,1))
#      }
# ]

def init_tests_lists():
    tests = []
    for kernel_size in [(2,2), (2,3), (3,2), (3,3)]:
        for strides in [(1,1), (2,2)]:
            for padding in ['valid', 'same']:
                for input_shape in [(6, 6, 1), (5, 5, 3)]:
                    name = f'Ker{kernel_size}_Str{strides}_Pad{padding[0].upper()}_Inp{input_shape}'
                    test = {'name': name,
                            'element': DepthwiseConv2D(kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           activation='linear',
                                           bias_initializer='random_normal',
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
