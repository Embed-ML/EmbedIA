from tensorflow.keras.layers import Dense


# Tests list
TESTS_LIST = [
    {'name': 'Unit50_ActLin_Imp50_BiasT',
     'element': Dense(units=50, activation='linear', input_dim=50, use_bias=True, bias_initializer='random_normal')}
]

if __name__ == '__main__':
    import sys

    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))