from tensorflow.keras.layers import ZeroPadding2D



# Tests list
TESTS_LIST = [
    {'name': 'Pad2x2_Inp4x4x3',
     'element': ZeroPadding2D(padding=(2,2), input_shape=(4, 4, 3))
     },
    {'name': 'Pad1x2_Inp4x4x3',
     'element': ZeroPadding2D(padding=(1, 2), input_shape=(4, 4, 3))
     },
    {'name': 'Pad2x1_Inp4x4x3',
     'element': ZeroPadding2D(padding=(2, 1), input_shape=(4, 4, 3))
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