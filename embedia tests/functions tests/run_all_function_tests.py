import sys, os, importlib

sys.path.append('..')   # Add the parent directory, base folder for test modules
sys.path.append('../..')   # Add the parent directory, base folder for embedia modules

from common.tester import Tester, TestSummary

current_filename = os.path.basename(__file__)  # Get the name of the current file

all_tests_list = []  # Empty list to store all the tests
tester = Tester()

test_file_lists = os.listdir(".")  # Get a list of files (and folders)

for filename in test_file_lists:  # Iterate through the files
    if (filename != current_filename and filename.endswith('.py') ):  # Check if it's a test file (not the current file)
        module = importlib.import_module(filename[:-3])  # Dynamically import the test module
        if hasattr(module,  'TESTS_LIST'): # variable for tests in file
            all_tests_list.extend(module.TESTS_LIST)  # Add the tests from the module to the list


for test in all_tests_list:
    print(f'{test["element"].__class__.__name__}_{test["name"]}')
results = tester.run_tests(all_tests_list, verbose=True) # Run all the collected tests with verbose output

print(TestSummary(results))