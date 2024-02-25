from visualize import create_html


MODELS_FILES = ['model.tflite', 'full_model.tflite']

import os

print(os. getcwd())

for filename in MODELS_FILES:
    tflite_input = '%s/models/%s' % (os. getcwd(), filename)
    html_output = '%s/outputs/%s.html' % (os. getcwd(), filename)

    html = create_html(tflite_input)
    with open(html_output, "w") as output_file:
        output_file.write(html)
