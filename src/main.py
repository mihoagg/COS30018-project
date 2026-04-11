from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from image_processing import processor
from image_segmentation.segmentation import segment_digits
from data.mnist import load_mnist
from model.cnn import build_cnn_model
from model.multi_layer_perceptron import build_mlp_model


def data_loader(source="mnist"):
    if source == "mnist":
        return load_mnist()
    raise ValueError("source must be 'mnist'")


def prepare_data_and_model(model_type):
    (x_train, y_train), (x_test, y_test) = data_loader("mnist")

    # Normalize training data
    x_train = processor.normalize(x_train)
    x_test = processor.normalize(x_test)

    if model_type == "cnn":
        # Add channel dimension for CNN
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
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


def _prepare_model_batch(images, model_type="cnn"):
    """Prepares a batch of processed images for model input."""
    batch = np.stack(images, axis=0).astype("float32")

    if model_type == "cnn":
        return batch[..., np.newaxis]

    if model_type == "mlp":
        return batch

    raise ValueError("model_type must be 'mlp' or 'cnn'")


def preprocess_image_array(image_array, model_type="cnn"):
    """Full preprocessing pipeline: grayscale, segmentation, resize, normalize."""
    gray_image = processor.to_grayscale(np.asarray(image_array))
    gray_image = processor.invert_if_needed(gray_image)
    segments = segment_digits(gray_image)

    if not segments:
        # If no digits found, treat the whole image as a single digit, cleaning the background
        cleaned_image = processor.clean_background(gray_image)
        processed_images = [processor.normalize(processor.resize_to_mnist(cleaned_image))]
    else:
        # Process each detected digit segment
        processed_images = [
            processor.normalize(processor.resize_to_mnist(processor.clean_background(segment.image)))
            for segment in segments
        ]

    image_batch = _prepare_model_batch(processed_images, model_type=model_type)
    is_multi_digit = len(processed_images) > 1
    
    # Return processed images for preview (scaled back to 0-1 for display tools)
    preview_output = processed_images if is_multi_digit else processed_images[0]
    
    return image_batch, preview_output, is_multi_digit


def preprocess_image(image_path, model_type="cnn"):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return preprocess_image_array(image, model_type=model_type)


def predict_image(model, image_path, model_type="cnn"):
    image_batch, preview_image, is_multi_digit = preprocess_image(
        image_path, model_type
    )
    prediction = model.predict(image_batch, verbose=0)
    predicted_labels = np.argmax(prediction, axis=1).astype(int).tolist()
    confidences = np.max(prediction, axis=1).astype(float).tolist()

    if is_multi_digit:
        combined_label = "".join(str(label) for label in predicted_labels)
        return combined_label, confidences, preview_image, prediction

    return predicted_labels[0], confidences[0], preview_image, prediction[0]


def predict_image_array(model, image_array, model_type="cnn"):
    image_batch, preview_image, is_multi_digit = preprocess_image_array(
        image_array, model_type
    )
    prediction = model.predict(image_batch, verbose=0)
    predicted_labels = np.argmax(prediction, axis=1).astype(int).tolist()
    confidences = np.max(prediction, axis=1).astype(float).tolist()

    if is_multi_digit:
        combined_label = "".join(str(label) for label in predicted_labels)
        return combined_label, confidences, preview_image, prediction

    return predicted_labels[0], confidences[0], preview_image, prediction[0]


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
        if isinstance(confidence, list):
            print(
                "Per-digit confidence:",
                ", ".join(f"{value:.4f}" for value in confidence),
            )
            figure, axes = plt.subplots(
                1,
                len(preview_image),
                figsize=(3 * len(preview_image), 3),
            )
            axes = np.atleast_1d(axes)
            for index, (axis, digit_image) in enumerate(
                zip(axes, preview_image),
                start=1,
            ):
                axis.imshow(digit_image, cmap="gray")
                axis.set_title(f"Digit {index}")
                axis.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Confidence: {confidence:.4f}")
            plt.imshow(preview_image, cmap="gray")
            plt.title(f"Predicted: {predicted_label}")
            plt.show()

    return model


if __name__ == "__main__":
    main()
