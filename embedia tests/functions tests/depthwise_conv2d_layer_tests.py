from tensorflow.keras.layers import DepthwiseConv2D

# Tests list
TESTS_LIST = [
    {'name': 'Ker3x3_Str1x1_Pad0_Mul1_Inp6x6x1', # simple test
     'element': DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                        input_shape=(6,6,1))
     },
     {'name': 'Ker3x3_Str1x1_Pad0_Mul1_Inp6x5x1', # test height > width
     'element': DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                        input_shape=(6,5,1))
     },
     {'name': 'Ker3x3_Str1x1_Pad0_Mul1_Inp5x6x1', # test width > height
     'element': DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                        input_shape=(5,6,1))
     },
     {'name': 'Ker2x2_Str1x1_Pad0_Mul1_Inp6x6x1', # first test with even kernel
     'element': DepthwiseConv2D(kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
                        input_shape=(6, 6, 1))
     },
     {'name': 'Ker2x2_Str1x1_Pad0_Mul1_Inp6x6x3', # first test with channels > 1
     'element': DepthwiseConv2D(kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
                        input_shape=(6, 6, 3))
     }
]

if __name__ == '__main__':
    import sys
    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))
