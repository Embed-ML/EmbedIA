from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def init(scaler, count):
    pass
# Tests list
TESTS_LIST = [
    # {'name': 'Basic',
    #  'element': StandardScaler(),
    #  'shape': (5,5)
    #  },
    # {'name': 'Basic',
    #  'element': MinMaxScaler(),
    #  'shape': (5,5)
    #  },
    {'name': 'Basic',
     'element': MaxAbsScaler(),
     'shape': (5,5)
     }
    # ,
    # {'name': 'Basic',
    #  'element': RobustScaler(),
    #  'shape': (5, 5)
    #  }

]

if __name__ == '__main__':
    import sys
    sys.path.append('..')  # embedia test folder
    sys.path.append('../..')  # embedia root folder

    from common.tester import Tester, TestSummary

    tester = Tester()
    results = tester.run_tests(TESTS_LIST, verbose=True)

    print(TestSummary(results))