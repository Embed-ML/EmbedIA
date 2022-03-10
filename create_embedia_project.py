from xml.etree.ElementTree import Comment

from sklearn.datasets import load_digits

from embedia.project_options import *
from embedia.project_generator import ProjectGenerator


############# Settings to create the project #############

OUTPUT_FOLDER = 'outputs'
PROJECT_NAME  = 'C_mnist_min_tanh_float'
MODEL_FILE    = 'models/mnist_model_min_tanh.h5'

model = load_model(MODEL_FILE)



options = ProjectOptions()

# options.project_type = ProjectType.ARDUINO
options.project_type = ProjectType.C
# options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
# options.data_type = ModelDataType.FIXED8

options.debug_mode = DebugMode.DISCARD
# options.debug_mode = DebugMode.DISABLED
# options.debug_mode = DebugMode.HEADERS
# options.debug_mode = DebugMode.DATA

'''
In case you want to use a specific example data, you can use the following code for example for MNIST dataset:
from tensorflow.keras.models import load_model
digits = load_digits()
example_number = 33
sample = digits.images[example_number] #this should be set to a sample of the data
comment= "number %d example for test" % digits.target[example_number] #this should be set to a comment about the data
options.example_data = sample
options.example_comment = comment
'''


options.example_data = None
options.example_comment = ''

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}


############# Generate project #############

generator = ProjectGenerator()
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)
#print("\n"+comment)