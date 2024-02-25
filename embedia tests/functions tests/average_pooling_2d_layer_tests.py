from tensorflow.keras.layers import AveragePooling2D


# Tests list
TESTS_LIST = [
    {'name': 'Size2_Str1_Pad0_Inp6x6x1',
     'element': AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', input_shape=(6,6,1))
    },
    {'name': 'Size2_Str1_Pad0_Inp5x6x1',
     'element': AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', input_shape=(5, 6, 1))
     },
    {'name': 'Size2_Str1_Pad0_Inp6x5x1',
     'element': AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', input_shape=(6, 5, 1))
     },
    {'name': 'Size2_Str1_Pad0_Inp6x6x3',
     'element': AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', input_shape=(6, 6, 3))
     },
    {'name': 'Size3_Str1_Pad0_Inp5x5x3',
     'element': AveragePooling2D(pool_size=(3, 3), strides=1, padding='valid', input_shape=(5, 5, 3))
     },
    {'name': 'Size3_Str1_Pad0_Inp5x5x3',
     'element': AveragePooling2D(pool_size=(3, 3), strides=1, padding='valid', input_shape=(5, 5, 3))
     },
    {'name': 'Size3_Str2_Pad0_Inp5x5x3',
     'element': AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=(5, 5, 3))
     },
    {'name': 'Size3_Str3_Pad0_Inp5x5x3',
     'element': AveragePooling2D(pool_size=(3, 3), strides=3, padding='valid', input_shape=(5, 5, 3))
     }

]

if __name__ == '__main__':
    import sys
    sys.path.append('..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))