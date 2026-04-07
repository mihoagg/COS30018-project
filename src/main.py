import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from src.image_processing.normalize_mnist import normalize
from src.model.multi_layer_perceptron import multi_layer_perceptron
from src.data.mnist import load_mnist

# Data loader
def data_loader(source='mnist'):
    if source == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist()
        return (x_train, y_train), (x_test, y_test)

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = data_loader('mnist')

# Normalize
x_train, x_test = normalize(x_train, x_test)

# Show an example image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Build model
model = multi_layer_perceptron()

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)
