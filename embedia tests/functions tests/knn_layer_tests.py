from sklearn.neighbors import KNeighborsClassifier


def init(scaler, count):
    pass


# Tests list
TESTS_LIST = [
    {'name': 'KNN-Cosine',
     'element': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
     'shape': (None, 10)
     },

    {'name': 'KNN-Euclidean',
     'element': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
     'shape': (None,10)
     },
    {'name': 'KNN-Manhattan',
     'element': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
     'shape': (None, 10)
     },
    {'name': 'KNN-Cosine',
     'element': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
     'shape': (None, 10)
     },
    {'name': 'KNN-Chebyshev',
     'element': KNeighborsClassifier(n_neighbors=5, metric='chebyshev'),
     'shape': (None, 10)
     },
    {'name': 'KNN-Bray-Curtis',
     'element': KNeighborsClassifier(n_neighbors=5, metric='braycurtis'),
     'shape': (None, 10)
     },
    {'name': 'KNN-Canberra',
     'element': KNeighborsClassifier(n_neighbors=5, metric='canberra'),
     'shape': (None, 10)
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