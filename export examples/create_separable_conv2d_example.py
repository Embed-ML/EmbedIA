import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')

from embedia.project_generator import ProjectGenerator
from embedia.model_generator.project_options import (
    ModelDataType,
    DebugMode,
    ProjectFiles,
    ProjectOptions,
    ProjectType
)

# Configuration
OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME = 'MNIST_SeparableConv2D'

# Load and preprocess MNIST data
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension for grayscale
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    
    # Select subset for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    # Select 10 test samples for export
    X_export = X_test[:10]
    y_export = y_test[:10]
    
    return X_train, y_train, X_export, y_export

# Create model with SeparableConv2D
def create_model():
    model = Sequential([
        SeparableConv2D(16, (3, 3), input_shape=(28, 28, 1)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        SeparableConv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Load MNIST data
print("Loading MNIST dataset...")
X_train, y_train, X_export, y_export = load_mnist_data()

# Create and train model
print("Creating MNIST model with SeparableConv2D layers...")
model = create_model()
model._name = "mnist_separable_conv2d"

print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

print("\nModel summary:")
model.summary()

# Evaluate model
test_loss, test_acc = model.evaluate(X_export, y_export, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Configure EmbedIA options
options = ProjectOptions()
options.embedia_folder = '../embedia/'
options.project_type = ProjectType.CODEBLOCK
options.data_type = ModelDataType.FLOAT
options.debug_mode = DebugMode.HEADERS
options.example_data = X_export
options.example_ids = y_export
options.files = ProjectFiles.ALL
options.clean_output = True

# Generate project
print("\nGenerating EmbedIA project...")
generator = ProjectGenerator(options)
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, options)

print(f"\nProject '{PROJECT_NAME}' exported successfully in '{OUTPUT_FOLDER}'")
print("The project includes:")
print("- MNIST digit classification model")
print("- SeparableConv2D layers for efficient computation")
print("- 10 MNIST test samples with labels")
print("- Complete C implementation for microcontrollers")

# Show example usage
print("\n=== Example C usage ===")
print("#include \"model.h\"")
print("")
print("// Initialize model")
print("model_init();")
print("")
print("// Run inference")
print("float input[784];  // 28x28 flattened image")
print("float results[10]; // 10 digit classes")
print("int prediction = model_predict(input, results);")
print("printf(\"Predicted digit: %d\\n\", prediction);")

print(model.evaluate(X_export, y_export, verbose=1))