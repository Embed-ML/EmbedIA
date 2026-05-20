import sys
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)
from embedia.utils.model_loader import ModelLoader


# Cargar y preprocesar la imagen
def load_and_preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')  # Forzar RGB
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    #img_array = np.expand_dims(img_array, axis=0)

    # Normalización para MobileNetV1 (entrada 0-255)
    img_array = img_array / 127.5 - 1.0  # Escala de -1 a 1

    return img_array


def decode_predictions(preds, top=5):
    """
    Decodifica predicciones de un modelo para ImageNet, soportando batch de ejemplos.

    Args:
        preds: Array de numpy con las predicciones. Puede ser:
               - 1D (un solo ejemplo, shape [1000] o [1001])
               - 2D (batch de ejemplos, shape [batch_size, 1000] o [batch_size, 1001])
        top: Número de predicciones top a devolver por ejemplo

    Returns:
        Lista de listas de tuplas (clase, probabilidad) para cada ejemplo
    """
    # Si es un solo ejemplo, convertirlo a batch de 1 ejemplo para procesamiento uniforme
    if len(preds.shape) == 1:
        preds = preds[np.newaxis, :]

    # Eliminar clase background si existe (1001 clases)
    if preds.shape[1] == 1001:
        preds = preds[:, 1:]

    # Obtener top índices para cada ejemplo
    top_indices = np.argsort(preds, axis=1)[:, -top:][:, ::-1]

    # Cargar etiquetas de ImageNet
    with open('../models/imagenet1000_clsidx_to_labels.es.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Procesar cada ejemplo en el batch
    results = []
    for example_idx in range(preds.shape[0]):
        example_results = [(classes[i], preds[example_idx, i]) for i in top_indices[example_idx]]
        results.append(example_results)

    return results

def debug_model_layers(model, sample, print_shape_only=False, max_values=5, print_weights=False):
    """
    Muestra las salidas de cada capa del modelo para una muestra dada, incluyendo pesos si se solicita.

    Args:
        model: Modelo de TensorFlow/Keras
        sample: Imagen de entrada (debe tener las dimensiones correctas)
        print_shape_only: Si True, solo imprime las formas de los tensores
        max_values: Número máximo de valores a mostrar si print_shape_only=False
        print_weights: Si True, muestra información sobre los pesos de cada capa
    """


    # Crear modelo que devuelva todas las salidas intermedias
    layer_outputs = [layer.output for layer in model.layers]
    debug_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

    # Obtener todas las salidas
    outputs = debug_model.predict(sample)

    print("\n=== DEBUG DE CAPAS Y PESOS ===")
    print(f"Input shape: {sample.shape}")

    for i, (layer, output) in enumerate(zip(model.layers, outputs)):
        print(f"\nLayer {i}: {layer.name} ({layer.__class__.__name__})")
        print(f"Output shape: {output.shape}")

        if not print_shape_only:
            # Para no saturar la salida, mostramos valores representativos
            flat_output = output.flatten()
            print(f"First {max_values} values: {flat_output[:max_values]}")
            if len(flat_output) > max_values:
                print(f" Last {max_values} values: {flat_output[-max_values:]}")

            # Estadísticas útiles
            print(f"Min...: {np.min(output):.4f}, Max...: {np.max(output):.4f}")
            print(f"Mean..: {np.mean(output):.4f}, Std...: {np.std(output):.4f}")

        if print_weights and hasattr(layer, 'weights') and layer.weights:
            print("\nWeights info:")
            for j, weight in enumerate(layer.weights):
                weight_name = weight.name.split('/')[-1]
                weight_values = weight.numpy()
                print(f"  Weight {j} ({weight_name}): shape {weight_values.shape}")

                if not print_shape_only:
                    flat_weights = weight_values.flatten()
                    print(f"    First {max_values} values: {flat_weights[:max_values]}")
                    if len(flat_weights) > max_values:
                        print(f"     Last {max_values} values: {flat_weights[-max_values:]}")

                    print(f"    Min...: {np.min(weight_values):.4f}, Max...: {np.max(weight_values):.4f}")
                    print(f"    Mean..: {np.mean(weight_values):.4f}, Std...: {np.std(weight_values):.4f}")

OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'Prj-MobileNetV1'
MODEL_FILE = '../models/mobilenet_v1_0.25_128_quant.tflite'
IMAGE_FILE1 = 'samples/elephant1.png'  # Asegúrate de tener esta imagen en tu directorio
IMAGE_FILE2 = 'samples/car1.png'  # Asegúrate de tener esta imagen en tu directorio

# Cargar el modelo
model = ModelLoader.load_model(MODEL_FILE)
model._name = "mobilenet_v1"

# Mostrar resumen del modelo
model.summary()


# Procesar imagen
input_size = (128, 128)  # Tamaño esperado por MobileNetV1
sample1 = load_and_preprocess_image(IMAGE_FILE1, input_size)
sample2 = load_and_preprocess_image(IMAGE_FILE2, input_size)
samples = np.array([sample1])#, sample2])

# Realizar predicción
predictions = model.predict(samples)

# debug_model_layers(model, sample1,  print_shape_only=False, max_values=6, print_weights=True)

# Mostrar resultados
tops = 5
decoded_preds = decode_predictions(predictions, tops)
# Mostrar cada ejemplo
for example_idx, example_preds in enumerate(decoded_preds):
    print(f"\nEjemplo {example_idx + 1} - Top {len(example_preds)} predicciones:")

    for i, (class_name, prob) in enumerate(example_preds[:tops]):
        print(f"  {i + 1}: {class_name} ({prob * 100:.2f}%)")

# Configuración del proyecto EmbedIA
options = ProjectOptions()
options.embedia_folder = '../embedia/'
options.project_type = ProjectType.CODEBLOCK
options.data_type = ModelDataType.QUANT8
options.debug_mode = DebugMode.DISABLED

# Usar la imagen de ejemplo para la generación del proyecto
options.example_data = samples
options.example_ids = np.argmax(predictions, axis=1) # IDs de la clase predicha

options.files = ProjectFiles.ALL
options.clean_output = True

############# Generar proyecto #############
generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print(f"\nProyecto {PROJECT_NAME} exportado en {OUTPUT_FOLDER}")