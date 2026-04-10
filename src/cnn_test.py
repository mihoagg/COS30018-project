import matplotlib.pyplot as plt

from src.data.mnist import load_mnist
from src.image_processing.cnn_preprocessing import cnn_preprocessing
from src.image_processing.normalize_mnist import normalize
from src.model.cnn import build_cnn_model


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()

    x_train, x_test = normalize(x_train, x_test)
    x_train, x_test = cnn_preprocessing(x_train, x_test)

    plt.imshow(x_train[0].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[0]}")
    plt.show()

    model = build_cnn_model()
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
