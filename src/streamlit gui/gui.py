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

from data.emnist import get_label_mapping
from main import (
    data_loader,
    evaluate_model,
    load_trained_model,
    predict_image_array,
    save_model,
    train_model,
)


st.set_page_config(page_title="Handwritten Character Classifier", layout="wide")
st.title("Handwritten Digit & Letter Classifier")
st.caption("Train model, upload image, and inspect prediction confidence.")


def get_model_type_label(model_type):
    labels = {"cnn": "CNN", "mlp": "MLP", "svm": "SVM"}
    return labels.get(model_type, model_type.upper())


def show_history(history):
    if history is None:
        st.info("No training history available for this model type (e.g., SVM).")
        return
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


def show_probability_chart(probabilities, source="mnist"):
    mapping = get_label_mapping(source)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(probabilities))
    ax.bar(x, probabilities)
    ax.set_xticks(x)
    ax.set_xticklabels(mapping, rotation=45 if len(mapping) > 10 else 0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Character")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
    plt.close(fig)


def show_classification_report(report, source="mnist"):
    if not report:
        return
    
    st.write("### Per-class Metrics (Test Set)")
    
    mapping = get_label_mapping(source)
    table_data = []
    for i, label in enumerate(mapping):
        str_idx = str(i)
        if str_idx in report:
            metrics = report[str_idx]
            table_data.append({
                "Character": label,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1-score']:.4f}",
                "Support": int(metrics['support'])
            })
            
    st.table(table_data)


def show_uploaded_prediction(predicted_label, confidence, processed_image, probabilities, source="mnist"):
    preview_col, result_col = st.columns([1, 1.2])

    with preview_col:
        if isinstance(processed_image, list):
            st.write("Processed Characters")
            preview_columns = st.columns(len(processed_image))
            for index, (column, digit_image) in enumerate(
                zip(preview_columns, processed_image),
                start=1,
            ):
                column.image(
                    digit_image,
                    caption=f"Char {index}",
                    use_container_width=True,
                    clamp=True,
                )
        else:
            st.image(
                processed_image,
                caption="Processed Image",
                width=220,
                clamp=True,
            )

    with result_col:
        st.write(f"Predicted Label: `{predicted_label}`")
        if isinstance(confidence, list):
            st.write(
                "Per-character confidence: "
                + ", ".join(f"`{value:.4f}`" for value in confidence)
            )
            for index, char_probabilities in enumerate(probabilities, start=1):
                st.caption(f"Character {index} probabilities")
                show_probability_chart(char_probabilities, source=source)
        else:
            st.write(f"Confidence: `{confidence:.4f}`")
            show_probability_chart(probabilities, source=source)


def ensure_state():
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("model_type", "cnn")
    st.session_state.setdefault("source", "mnist")
    st.session_state.setdefault("history", None)
    st.session_state.setdefault("metrics", None)


ensure_state()

with st.sidebar:
    st.header("Settings")
    source = st.selectbox(
        "Dataset",
        options=["mnist", "emnist"],
        format_func=lambda x: "Digits (MNIST)" if x == "mnist" else "Digits & Letters (EMNIST)",
        index=0 if st.session_state["source"] == "mnist" else 1
    )
    st.session_state["source"] = source
    
    (x_train, y_train), (x_test, y_test) = data_loader(source)

    st.header("Training Controls")
    model_type = st.selectbox(
        "Model Type",
        options=["cnn", "mlp", "svm"],
        format_func=get_model_type_label,
    )
    
    model_params = {}
    epochs = 5
    batch_size = 32
    optimizer = "adam"
    test_size = 0.1

    if model_type in ["cnn", "mlp"]:
        epochs = st.slider("Epochs", min_value=1, max_value=15, value=5)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=0)
        
        st.subheader("Architecture")
        dense_units = st.slider("Dense Units", min_value=32, max_value=256, value=128, step=32)
        if model_type == "cnn":
            conv1_filters = st.slider("Conv Layer 1 Filters", min_value=16, max_value=64, value=32, step=16)
            conv2_filters = st.slider("Conv Layer 2 Filters", min_value=32, max_value=128, value=64, step=32)
            model_params = {
                "conv1_filters": conv1_filters,
                "conv2_filters": conv2_filters,
                "dense_units": dense_units,
            }
        else:
            model_params = {"dense_units": dense_units}
    elif model_type == "svm":
        test_size = st.slider(
            "Training Set Size Fraction",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            help="Fraction of training data to use for SVM (it is slow on full dataset).",
        )
        svm_c = st.number_input("C (Regularization)", value=1.0, step=0.1)
        svm_kernel = st.selectbox("Kernel", options=["rbf", "linear", "poly", "sigmoid"], index=0)
        svm_gamma = st.selectbox("Gamma", options=["scale", "auto"], index=0)
        model_params = {"C": svm_c, "kernel": svm_kernel, "gamma": svm_gamma}

    default_ext = ".joblib" if model_type == "svm" else ".keras"
    model_save_path = st.text_input(
        "Save Model Path", 
        value=f"trained_models/{model_type}_{source}_model{default_ext}"
    )
    model_load_path = st.text_input(
        "Load Model Path", 
        value=f"trained_models/{model_type}_{source}_model{default_ext}"
    )

    if st.button("Train Model", use_container_width=True):
        with st.spinner(f"Training model on {source}..."):
            model, _, test_loss, test_acc, history, duration, report = train_model(
                model_type=model_type,
                source=source,
                epochs=epochs,
                optimizer=optimizer,
                batch_size=batch_size,
                verbose=0,
                test_size=test_size,
                **model_params
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
            "training_duration": float(duration),
            "report": report,
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
            with st.spinner("Evaluating loaded model..."):
                report = evaluate_model(loaded_model, model_type, x_test, y_test)
                # Accuracy is in the report
                test_acc = report["accuracy"]
        except Exception as error:
            st.error(f"Load failed: {error}")
        else:
            st.session_state["model"] = loaded_model
            st.session_state["model_type"] = model_type
            st.session_state["history"] = None
            st.session_state["metrics"] = {
                "test_accuracy": float(test_acc),
                "test_loss": 0.0,
                "epochs": "N/A",
                "batch_size": "N/A",
                "optimizer": "N/A",
                "training_duration": 0.0,
                "report": report,
            }
            st.success("Model loaded and evaluated.")

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Model Status")
    if st.session_state["model"] is None:
        st.info("No model ready. Train or load one from sidebar.")
    else:
        st.write(f"Current model: `{get_model_type_label(st.session_state['model_type'])}`")
        metrics = st.session_state["metrics"]
        if metrics:
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
            metric_col2.metric("Test Loss", f"{metrics['test_loss']:.4f}")
            metric_col3.metric("Batch Size", metrics["batch_size"])
            metric_col4.metric("Training Time", f"{metrics['training_duration']:.2f}s")
            st.write(
                f"Optimizer: `{metrics['optimizer']}` | Epochs: `{metrics['epochs']}`"
            )
            show_classification_report(metrics.get("report"), source=st.session_state["source"])
        else:
            st.caption("Loaded model. Training metrics unavailable.")

with right_col:
    st.subheader(f"{source.upper()} Sample Test")
    sample_index = st.slider("Test Sample Index", 0, len(x_test) - 1, 0)
    sample_image = x_test[sample_index]
    
    mapping = get_label_mapping(source)
    actual_label = mapping[int(y_test[sample_index])]
    
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
            show_probability_chart(probabilities, source=st.session_state["source"])

st.subheader("Upload Your Image")
uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("L")
    uploaded_array = np.array(uploaded_image)
    
    # Display columns for original and thresholded view
    input_col, threshold_col = st.columns(2)
    with input_col:
        st.image(uploaded_array, caption="Uploaded Image", width=220, clamp=True)
    
    with threshold_col:
        from image_processing.processor import threshold_image
        binary_view = threshold_image(uploaded_array)
        #st.image(binary_view, caption="Processed image", width=220, clamp=True)

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
            source=st.session_state["source"]
        )
