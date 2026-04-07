import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Show an example image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # turns image to list of num
    layers.Dense(128, activation='relu'), # learns pattern
    layers.Dense(10, activation='softmax') # outputs probability for digits 0–9
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

print(x_train[0])

#print(y_train[0])
print(y_test[0])

# Pick a test image
index = 0
image = x_test[index]

# Get prediction (output layer values)
prediction = model.predict(image.reshape(1, 28, 28))

print(prediction)

"""Test Prediction"""

import numpy as np

# Pick a random test image
index = 0
image = x_test[index]

# Predict
prediction = model.predict(image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.show()

