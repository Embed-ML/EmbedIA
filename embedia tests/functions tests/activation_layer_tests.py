from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, tanh, sigmoid


# Tests list
TESTS_LIST = [
    {'name': 'Linear',
     'element': Activation('linear', input_shape=(1,10))
     },
    {'name': 'Softmax',
     'element': Activation('softmax', input_shape=(1,10))
     },
    {'name': 'Softsign',
     'element': Activation('softsign', input_shape=(1,10))
     },
    {'name': 'reLU',
     'element': Activation(relu, input_shape=(1,10))
     },
    {'name': 'LeakyReLU',
     'element': Activation('LeakyReLU',input_shape=(1, 10))
     },
    {'name': 'Tanh',
     'element': Activation(tanh, input_shape=(1, 10))
     },
    {'name': 'Sigmoid',
     'element': Activation(sigmoid, input_shape=(1, 10))
     }
]

if __name__ == '__main__':
    import sys
    sys.path.append('..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))