from tensorflow.keras.layers import Conv2D

# Tests list
TESTS_LIST = [
    {'name': 'Ker3x3_Str1x1_Pad0_Inp8x8x1', # simple test
     'element': Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                       input_shape=(8,8,1))
     },
     {'name': 'Ker3x3_Str1x1_Pad0_Inp8x5x1', # test height > width
     'element': Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                       input_shape=(8,5,1))
     },
     {'name': 'Ker3x3_Str1x1_Pad0_Inp5x8x1', # test width > height
     'element': Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear',
                       input_shape=(5,8,1))
     },
     {'name': 'Ker2x2_Str1x1_Pad0_Inp5x8x1', # first test with even kernel
     'element': Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
                       input_shape=(8, 8, 1))
     },
     {'name': 'Ker2x2_Str1x1_Pad0_Inp8x8x3', # first test with channels > 1
     'element': Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='linear',
                       input_shape=(8, 8, 3))
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
