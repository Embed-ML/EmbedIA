import sys
import os
import re

sys.path.append('..')  # embedia root folder

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from common.data_generator import *
from common.code_generator import CodeGenerator
from common.compilers import GccCompiler
from embedia.model_generator.project_options import ModelDataType


# test
input_size = 50
output_size = 50

# C Code Generator
embedia_path = '../../embedia/'
output_path = '/'
datatype = ModelDataType.FIXED16
#datatype = ModelDataType.FLOAT

generator = CodeGenerator(embedia_path, output_path)

# Tests list
tests = [
    {'name':  'Basic',
     'element': Dense(units=output_size, activation='linear', input_dim=input_size, use_bias=True)}
    ]

tester = TFLayerDataGenerator()
tester.generate_inputs((1,input_size))
for test in tests:
    test_name = test['name']
    tester.test_element = test['element']
    tester.test()
    # print("Entradas:", tester.input_data)
    #print(" Salida ", tester.output_data)

    project_name = f'{ModelDataType.get_name(datatype)}_{tester.test_element.__class__.__name__}_{test_name}'
    generator.set_embedia_type(datatype)
    generator.set_project_name(project_name)

    generator.generate(tester.model, input=tester.input_data, output=tester.output_data, error_bound=0.2)

    ######################## COMPILER ###########################
    compiler = GccCompiler()

    (result, output_str) = compiler.compile(generator.get_project_folder(), 'main.c', generator.get_filenames())

    if result == 0:
        result = compiler.run(os.path.join(generator.get_project_folder(), 'main.exe'))
        print('Run result', result)

        # Utilizar expresión regular para encontrar el número
        match = re.search(r'result:\s*(\d+\.\d+)', result[1])

        if match:
            result_number = float(match.group(1))
            print("Resultado del test:", result_number)
        else:
            print("Error en el test")
    else:
        print('Compile error', output_str)



