import os, shutil
import re
from common.data_generator import *
from common.code_generator import CodeGenerator
from common.compilers import GccCompiler
from embedia.model_generator.project_options import ModelDataType
from embedia.utils.model_inspector import ModelInspector
from enum import Enum, auto
from common.tests_config import TestsConfig
from prettytable import PrettyTable


class TestResult(Enum):
    COMPILE_ERROR = auto()  # error compiling test
    EXECUTE_ERROR = auto()  # error executing test
    TEST_ERROR = auto()  # error running test
    TEST_FAILURE = auto()  # test doesn't achieve 100%
    TEST_BORDERLINE = auto()  # test achieve between 80% and 89.99%
    TEST_ACCEPTABLE = auto()  # test achieve between 90% and 99.99%
    TEST_SUCCESS = auto()  # test achieves 100%

    def simple_name(self):
        return self.name.lower().replace('test_', '').replace('_', ' ')

    def clr(self, text):
        result_colors = {
            TestResult.COMPILE_ERROR: '\033[91m',   # Rojo para COMPILE_ERROR
            TestResult.EXECUTE_ERROR: '\033[91m',   # Rojo para EXECUTE_ERROR
            TestResult.TEST_ERROR: '\033[91m',      # Rojo para TEST_ERROR
            TestResult.TEST_FAILURE: '\033[91m',    # Rojo para TEST_FAILURE
            TestResult.TEST_BORDERLINE: '\033[33m', # Amarillo para TEST_BORDERLINE
            TestResult.TEST_ACCEPTABLE: '\033[93m', # Amarillo para TEST_ACCEPTABLE
            TestResult.TEST_SUCCESS: '\033[92m',    # Verde para TEST_SUCCESS
            'reset': '\033[0m',
        }

        color_code = result_colors[self]
        formatted_text = f'{color_code}{text}\033[0m'

        return formatted_text

class TestSummary:
    _summary_by_classname = {}
    _summary_by_datatype ={}
    _total_tests = 0
    _results = []
    def __init__(self, results=None):
        if results is not None:
            self._results = results
            self._generate_summary()

    def __str__(self):
        return self.as_table()

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value
        self._generate_summary()

    def _add(self, tested_class: str, datatype: ModelDataType, result: TestResult):
        # summary by class & datatype
        key = (tested_class, datatype)
        if key not in self._summary_by_classname:
            self._summary_by_classname[key] = {k:0 for k in TestResult}

        self._summary_by_classname[key][result]+=1

        # summary by datatype

        if datatype not in self._summary_by_datatype:
            self._summary_by_datatype[datatype] = {k:0 for k in TestResult}

        self._summary_by_datatype[datatype][result] += 1
        self._total_tests += 1



    def _generate_summary(self):
        self._summary_by_classname = {}
        for result in self.results:
            cls_name = result['test_instance'].__class__.__name__
            self._add(cls_name, result['datatype'], result['state'])

    def as_table(self):
        result = f'\nSummary by Tested Module & Data Type of {self._total_tests} Tests\n'
        result += self._classname_summary_as_table()+'\n\n'
        result += f'Summary by Data Type of {self._total_tests} Tests\n'
        result += self._datatype_summary_as_table()+'\n\n'
        return result
    def _classname_summary_as_table(self):
        table = PrettyTable()
        table.field_names = ['Test Class', 'Data Type', 'Errors', 'Failure', 'Borderline', 'Acceptable', 'Success']

        for (tested_class, class_data), results  in self._summary_by_classname.items():
            row = [tested_class, ModelDataType.get_name(class_data)]
            # group all errors
            errors = results[TestResult.COMPILE_ERROR] + results[TestResult.EXECUTE_ERROR]+ results[TestResult.TEST_ERROR]
            if errors > 0:
                row.append(TestResult(1).clr(errors))
            else:
                row.append('')

            for i in range(TestResult.TEST_FAILURE.value, TestResult.TEST_SUCCESS.value+1):
                count = results[TestResult(i)]
                if count == 0:
                    row.append('')
                else:
                    row.append(TestResult(i).clr(count))

            table.add_row(row)

        return str(table)

    def _datatype_summary_as_table(self):
        table = PrettyTable()
        table.field_names = ['Data Type', 'Errors', 'Failure', 'Borderline', 'Acceptable', 'Success']

        for datatype, results in self._summary_by_datatype.items():
            row = [ModelDataType.get_name(datatype)]
            # group all errors
            errors = results[TestResult.COMPILE_ERROR] + results[TestResult.EXECUTE_ERROR]+ results[TestResult.TEST_ERROR]
            if errors > 0:
                row.append(TestResult(1).clr(errors))
            else:
                row.append('')

            for i in range(TestResult.TEST_FAILURE.value, TestResult.TEST_SUCCESS.value+1):
                count = results[TestResult(i)]
                if count == 0:
                    row.append('')
                else:
                    row.append(TestResult(i).clr(count))


            table.add_row(row)

        return str(table)



class Tester:
    def __init__(self):
        self._config = TestsConfig()
        self._update()

    def _update(self):
        cfg = self._config
        self._code_generator = CodeGenerator(cfg.EMBEDIA_PATH, cfg.OUTPUT_PATH)
        self._data_generator = TFLayerDataGenerator()
        self._compiler = GccCompiler()

    def _inspect(self, model, input_data, filename):
        inspector = ModelInspector(model)
        try:
            inspector.save(filename, input_data, ln_break=-1)
        except:
            pass
    def _print_result(self, test_result, len_tname=80):
        out_line = test_result['test'][:len_tname].ljust(len_tname, '.') + ': '
        value = test_result['value']
        if isinstance(value, (int, float)) and value >= 0:
            value = '{:5.1f}%'.format(value)
            message = ''
        else:
            value = ' ' * 5
            message = ': ' + test_result['message']
        error = 'bound:{:5f}'.format(test_result['error_bound'])
        state = test_result['state']
        out_line += f'{value} {error} ({state.simple_name()}{message})'
        print(state.clr(out_line))

    def run_tests(self, tests_list, verbose=False):
        cfg = self._config
        code_gen = self._code_generator
        data_gen = self._data_generator
        compiler = self._compiler

        results = []
        # test
        for test in tests_list:
            test_name = test['name']
            test_elem = test['element']


            data_gen.test_element = test_elem
            if 'shape' in test:
                data_gen.generate(test['shape'])
            else:
                data_gen.generate()
            base_project_name = f'{data_gen.test_element.__class__.__name__}_{test_name}'

            for datatype in cfg.DATA_TYPES:

                error_bound = cfg.DATA_TYPE_BOUND_ERROR[datatype]
                project_name = f'{base_project_name}_{ModelDataType.get_name(datatype)}'
                test_result = {
                    'test': project_name,        # set name for result
                    'error_bound': error_bound,  # set error bound for test
                    'test_instance': test_elem,
                    'datatype': datatype
                }
                code_gen.set_embedia_type(datatype)
                code_gen.set_project_name(project_name)
                project_folder = code_gen.get_project_folder()
                code_gen.generate(data_gen.model, input=data_gen.input_data, output=data_gen.output_data,
                                  error_bound=error_bound)

                ######################## COMPILER ###########################
                (comp_result, output_str) = compiler.compile(project_folder, 'main.c',
                                                             code_gen.get_filenames())

                value = -1
                message = ''
                state = None

                if comp_result == 0:
                    (run_result, output_str) = compiler.run(os.path.join(project_folder, 'main.exe'))
                    # print(output_str)
                    if run_result == 0:
                        # Utilizar expresión regular para encontrar el número debe decir "result: xxx.xx%" en la salida
                        match = re.search(r'result:\s*(\d+\.\d+)', output_str)
                        if match:
                            value = float(match.group(1))
                            if value < 80.0:
                                state = TestResult.TEST_FAILURE
                                message = 'Test failure'
                            elif value < 90.0:
                                state = TestResult.TEST_BORDERLINE
                                message = 'Test in acceptable borderline'
                            elif value < 100.0:
                                state = TestResult.TEST_ACCEPTABLE
                                message = 'Test in acceptable'
                            else:
                                state = TestResult.TEST_SUCCESS
                                message = 'Test successfull'
                        else:
                            state = TestResult.TEST_ERROR
                            message = 'Test execution error.' + output_str
                    else:
                        state = TestResult.TEST_ERROR
                        message = 'Test execution error.' + output_str
                else:
                    state = TestResult.COMPILE_ERROR
                    message = 'Test compilation error.' + output_str

                test_result['message'] = message
                test_result['state'] = state
                test_result['value'] = value

                results.append(test_result)
                if verbose:
                    self._print_result(test_result)

                if not self._config.KEEP_SUCCESS_TESTS and state == TestResult.TEST_SUCCESS:
                    try:
                        shutil.rmtree(project_folder)
                    except:
                        pass
                else:
                    self._inspect(data_gen.model, data_gen.input_data, os.path.join(project_folder, 'computed_output.txt'))

        return results
