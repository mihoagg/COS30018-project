from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from image_processing.cnn_preprocessing import cnn_preprocessing
from image_processing.normalize_mnist import normalize
from data.mnist import load_mnist
from model.cnn import build_cnn_model
from model.multi_layer_perceptron import build_mlp_model


def data_loader(source="mnist"):
    if source == "mnist":
        return load_mnist()
    raise ValueError("source must be 'mnist'")


def prepare_data_and_model(model_type):
    (x_train, y_train), (x_test, y_test) = data_loader("mnist")

    x_train, x_test = normalize(x_train, x_test)

    if model_type == "cnn":
        x_train, x_test = cnn_preprocessing(x_train, x_test)
        model = build_cnn_model()
        display_image = x_train[0].squeeze()
    elif model_type == "mlp":
        model = build_mlp_model()
        display_image = x_train[0]
    else:
        raise ValueError("model_type must be 'mlp' or 'cnn'")

    return (x_train, y_train), (x_test, y_test), model, display_image


def compile_model(model, optimizer="adam"):
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model_type="cnn", epochs=5, optimizer="adam", batch_size=32, verbose=1):
    (x_train, y_train), (x_test, y_test), model, display_image = prepare_data_and_model(
        model_type
    )
    compile_model(model, optimizer=optimizer)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=verbose,
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)
    return model, display_image, test_loss, test_acc, history


def save_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path


def load_trained_model(model_path):
    return keras.models.load_model(model_path)


def preprocess_image_array(image_array, model_type="cnn"):
    resized_image = cv2.resize(image_array, (28, 28))
    normalized_image = resized_image.astype("float32") / 255.0

    if model_type == "cnn":
        return normalized_image.reshape(1, 28, 28, 1), normalized_image
    if model_type == "mlp":
        return normalized_image.reshape(1, 28, 28), normalized_image
    raise ValueError("model_type must be 'mlp' or 'cnn'")


def preprocess_image(image_path, model_type="cnn"):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return preprocess_image_array(image, model_type=model_type)


def predict_image(model, image_path, model_type="cnn"):
    image_batch, preview_image = preprocess_image(image_path, model_type)
    prediction = model.predict(image_batch, verbose=0)
    predicted_label = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction, axis=1)[0])
    return predicted_label, confidence, preview_image, prediction[0]


def predict_image_array(model, image_array, model_type="cnn"):
    image_batch, preview_image = preprocess_image_array(image_array, model_type)
    prediction = model.predict(image_batch, verbose=0)
    predicted_label = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction, axis=1)[0])
    return predicted_label, confidence, preview_image, prediction[0]


def main(
    model_type="cnn",
    epochs=5,
    image_path=None,
    model_output_path=None,
    trained_model_path=None,
):
    if trained_model_path:
        model = load_trained_model(trained_model_path)
        display_image = None
    else:
        model, display_image, _, test_acc, _ = train_model(
            model_type=model_type,
            epochs=epochs,
        )
        print("Test accuracy:", test_acc)

        if display_image is not None:
            plt.imshow(display_image, cmap="gray")
            plt.title("Training sample")
            plt.show()

        if model_output_path:
            saved_path = save_model(model, model_output_path)
            print(f"Model saved to: {saved_path}")

    if image_path:
        predicted_label, confidence, preview_image, _ = predict_image(
            model, image_path, model_type=model_type
        )
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        plt.imshow(preview_image, cmap="gray")
        plt.title(f"Predicted: {predicted_label}")
        plt.show()

    return model


if __name__ == "__main__":
    main()
