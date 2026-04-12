from tensorflow.keras import layers, models


def build_cnn_model(conv1_filters=32, conv2_filters=64, dense_units=128, num_classes=10):
    return models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(conv1_filters, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(conv2_filters, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
