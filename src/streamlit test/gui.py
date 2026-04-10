import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("MNIST Digit Classifier")

# Controls
epochs = st.slider("Epochs", 1, 10, 3)
model_type = st.selectbox("Model", ["Simple NN", "CNN"])

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model():
    model = models.Sequential()

    if model_type == "Simple NN":
        model.add(layers.Flatten(input_shape=(28, 28)))
        model.add(layers.Dense(128, activation='relu'))
    else:
        model.add(layers.Reshape((28,28,1), input_shape=(28,28)))
        model.add(layers.Conv2D(32, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train
if st.button("Train Model"):
    model = build_model()
    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=0)

    st.session_state["model"] = model

    # Plot accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='train')
    ax.plot(history.history['val_accuracy'], label='val')
    ax.legend()
    st.pyplot(fig)
    
st.subheader("Test on MNIST dataset")

if st.button("Test Random MNIST Image"):
    if "model" not in st.session_state:
        st.warning("Train the model first!")
    else:
        import random

        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        label = y_test[idx]

        # Show image
        st.image(img, width=150, caption=f"Actual Label: {label}")

        # Predict
        img_input = np.expand_dims(img, axis=0)
        pred = st.session_state["model"].predict(img_input)
        predicted_label = np.argmax(pred)

        st.write(f"Predicted: {predicted_label}")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image")

if uploaded_file and "model" in st.session_state:
    img = Image.open(uploaded_file).convert('L').resize((28,28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = st.session_state["model"].predict(img)
    st.write("Prediction:", np.argmax(pred))