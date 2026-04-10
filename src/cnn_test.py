import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from image_processing.cnn_preprocessing import cnn_preprocessing

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension (important for CNN)
x_train, x_test = cnn_preprocessing(x_train, x_test)

# Show an example image
plt.imshow(x_train[0].squeeze(), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
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