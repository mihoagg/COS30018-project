from tensorflow.keras import layers, models


def build_mlp_model(dense_units=128, num_classes=10):
    return models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
