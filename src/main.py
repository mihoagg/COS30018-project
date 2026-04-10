import matplotlib.pyplot as plt

from image_processing.cnn_preprocessing import cnn_preprocessing
from image_processing.normalize_mnist import normalize
from data.mnist import load_mnist
from model.cnn import build_cnn_model
from model.multi_layer_perceptron import build_mlp_model


def data_loader(source="mnist"):
    if source == "mnist":
        return load_mnist()


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


def main(model_type="cnn"):
    (x_train, y_train), (x_test, y_test), model, display_image = prepare_data_and_model(
        model_type
    )

    plt.imshow(display_image, cmap="gray")
    plt.title(f"Label: {y_train[0]}")
    plt.show()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)


if __name__ == "__main__":
    main()
