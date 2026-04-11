from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.mnist import load_mnist
from main import (
    load_trained_model,
    predict_image_array,
    save_model,
    train_model,
)


st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
st.title("MNIST Handwritten Digit Classifier")
st.caption("Train model, upload digit image, inspect prediction confidence.")


def get_model_type_label(model_type):
    return "CNN" if model_type == "cnn" else "MLP"


def show_history(history):
    history_data = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_data["accuracy"], label="Train Accuracy")
    axes[0].plot(history_data["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_data["loss"], label="Train Loss")
    axes[1].plot(history_data["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    st.pyplot(fig)
    plt.close(fig)


def show_probability_chart(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    digits = np.arange(10)
    ax.bar(digits, probabilities)
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
    plt.close(fig)


def show_uploaded_prediction(predicted_label, confidence, processed_image, probabilities):
    preview_col, result_col = st.columns([1, 1.2])

    with preview_col:
        if isinstance(processed_image, list):
            st.write("Processed Digits")
            preview_columns = st.columns(len(processed_image))
            for index, (column, digit_image) in enumerate(
                zip(preview_columns, processed_image),
                start=1,
            ):
                column.image(
                    digit_image,
                    caption=f"Digit {index}",
                    use_container_width=True,
                    clamp=True,
                )
        else:
            st.image(
                processed_image,
                caption="Processed 28x28 Input",
                width=220,
                clamp=True,
            )

    with result_col:
        st.write(f"Predicted Label: `{predicted_label}`")
        if isinstance(confidence, list):
            st.write(
                "Per-digit confidence: "
                + ", ".join(f"`{value:.4f}`" for value in confidence)
            )
            for index, digit_probabilities in enumerate(probabilities, start=1):
                st.caption(f"Digit {index} probabilities")
                show_probability_chart(digit_probabilities)
        else:
            st.write(f"Confidence: `{confidence:.4f}`")
            show_probability_chart(probabilities)


def ensure_state():
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("model_type", "cnn")
    st.session_state.setdefault("history", None)
    st.session_state.setdefault("metrics", None)


ensure_state()

(x_train, y_train), (x_test, y_test) = load_mnist()

with st.sidebar:
    st.header("Training Controls")
    model_type = st.selectbox(
        "Model Type",
        options=["cnn", "mlp"],
        format_func=get_model_type_label,
    )
    epochs = st.slider("Epochs", min_value=1, max_value=15, value=5)
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
    optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=0)
    model_save_path = st.text_input("Save Model Path", value="trained_models/digit_model.keras")
    model_load_path = st.text_input("Load Model Path", value="trained_models/digit_model.keras")

    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            model, _, test_loss, test_acc, history = train_model(
                model_type=model_type,
                epochs=epochs,
                optimizer=optimizer,
                batch_size=batch_size,
                verbose=0,
            )

        st.session_state["model"] = model
        st.session_state["model_type"] = model_type
        st.session_state["history"] = history
        st.session_state["metrics"] = {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
        }
        st.success("Training complete.")

    if st.button("Save Current Model", use_container_width=True):
        if st.session_state["model"] is None:
            st.warning("Train or load model first.")
        else:
            save_path = save_model(st.session_state["model"], model_save_path)
            st.success(f"Saved to {save_path}")

    if st.button("Load Saved Model", use_container_width=True):
        try:
            loaded_model = load_trained_model(model_load_path)
        except Exception as error:
            st.error(f"Load failed: {error}")
        else:
            st.session_state["model"] = loaded_model
            st.session_state["model_type"] = model_type
            st.session_state["history"] = None
            st.session_state["metrics"] = None
            st.success("Model loaded.")

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Model Status")
    if st.session_state["model"] is None:
        st.info("No model ready. Train or load one from sidebar.")
    else:
        st.write(f"Current model: `{get_model_type_label(st.session_state['model_type'])}`")
        metrics = st.session_state["metrics"]
        if metrics:
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
            metric_col2.metric("Test Loss", f"{metrics['test_loss']:.4f}")
            metric_col3.metric("Batch Size", metrics["batch_size"])
            st.write(
                f"Optimizer: `{metrics['optimizer']}` | Epochs: `{metrics['epochs']}`"
            )
        else:
            st.caption("Loaded model. Training metrics unavailable.")

    history = st.session_state["history"]
    # if history is not None:
    #     st.subheader("Training Curves")
    #     show_history(history)

with right_col:
    st.subheader("MNIST Sample Test")
    sample_index = st.slider("Test Sample Index", 0, len(x_test) - 1, 0)
    sample_image = x_test[sample_index]
    actual_label = int(y_test[sample_index])
    st.image(sample_image, caption=f"Actual Label: {actual_label}", width=220, clamp=True)

    if st.button("Predict Selected Test Sample"):
        if st.session_state["model"] is None:
            st.warning("Train or load model first.")
        else:
            predicted_label, confidence, _, probabilities = predict_image_array(
                st.session_state["model"],
                sample_image,
                model_type=st.session_state["model_type"],
)
            st.write(f"Predicted Label: `{predicted_label}`")
            st.write(f"Confidence: `{confidence:.4f}`")
            show_probability_chart(probabilities)

st.subheader("Upload Your Image")
uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("L")
    uploaded_array = np.array(uploaded_image)
    
    # Display columns for original and thresholded view
    input_col, threshold_col = st.columns(2)
    with input_col:
        st.image(uploaded_array, caption="Uploaded Image (Grayscale)", width=220, clamp=True)
    
    with threshold_col:
        from image_processing.processor import threshold_image
        binary_view = threshold_image(uploaded_array)
        st.image(binary_view, caption="Segmentation Mask (Binary)", width=220, clamp=True)

    if st.session_state["model"] is None:
        st.warning("Train or load model first.")
    else:
        predicted_label, confidence, processed_image, probabilities = predict_image_array(
            st.session_state["model"],
            uploaded_array,
            model_type=st.session_state["model_type"],
        )
        show_uploaded_prediction(
            predicted_label,
            confidence,
            processed_image,
            probabilities,
        )
