from tensorflow.keras import layers,models

multi_layer_perceptron = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # turns image to list of num
    layers.Dense(128, activation='relu'), # learns pattern
    layers.Dense(10, activation='softmax') # outputs probability for digits 0–9
])