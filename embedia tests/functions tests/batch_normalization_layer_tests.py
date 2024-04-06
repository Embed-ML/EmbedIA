from tensorflow.keras.layers import BatchNormalization


# Tests list
TESTS_LIST = [
    {'name': 'Basic',
     'element': BatchNormalization(input_shape=(2,2,3))}
]

if __name__ == '__main__':
    import sys

    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))