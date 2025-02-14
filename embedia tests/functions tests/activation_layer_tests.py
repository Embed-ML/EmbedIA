from tensorflow.keras.layers import Activation, Dense, ReLU, LeakyReLU, Softmax
from tensorflow.keras.activations import relu, tanh, sigmoid



def init_tests_lists():

    ACT_INFO = ['linear', 'ReLU', 'tanh', 'sigmoid', 'softmax', 'softsign', 'LeakyReLU']
    tests = []
    for act_name in ACT_INFO:
        tests.append(
            {'name': act_name+'_layer',
             'element': Activation(act_name, input_shape=(1, 10)) # Test activation as Layer
             }
        )
        tests.append(
            {'name': act_name+'_property',
             'element': Dense(units=10, input_dim=10, activation=act_name) # Test activation as property
             }
        )

    tests.append(
        {'name': 'specific_class',
         'element': ReLU(input_shape=(1, 10))  # Test activation as specific Layer
         }
    )
    tests.append(
        {'name': 'specific_class',
         'element': LeakyReLU(input_shape=(1, 10))  # Test activation as specific Layer
         }
    )
    tests.append(
        {'name': 'specific_class',
         'element': Softmax(input_shape=(1, 10))  # Test activation as specific Layer
         }
    )
    return tests


# Tests list
# TESTS_LIST = [
#     {'name': 'Linear',
#      'element': Activation('linear', input_shape=(1,10))
#      }
# ]
# TESTS_LIST = [
#     {'name': 'Linear',
#      'element': Dense(units=10, input_dim=10, activation='linear')
#      }
# ]

TESTS_LIST = init_tests_lists()

if __name__ == '__main__':
    import sys

    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))