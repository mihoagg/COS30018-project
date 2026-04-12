# Handwritten Character Recognition System

A modular machine learning application designed to recognize handwritten digits (MNIST) and alphabetical characters (EMNIST). The system features a custom image processing pipeline, multi-model support (CNN, MLP, SVM), and an interactive Streamlit GUI.

## 🚀 Features

- **Dual Dataset Support**: Switch between **MNIST** (10 digits) and **EMNIST Balanced** (47 classes: digits + uppercase/lowercase letters).
- **Multiple Architectures**:
  - **CNN**: Highly accurate Convolutional Neural Network with custom filter settings.
  - **MLP**: Simple Multi-Layer Perceptron for baseline testing.
  - **SVM**: Support Vector Machine classifier for traditional ML comparison.
- **Advanced Preprocessing**: Custom pipeline for grayscale conversion, Otsu thresholding, noise reduction, and centered resizing.
- **Digit Segmentation**: Automatic detection and segmentation of multiple characters within a single uploaded image.
- **Interactive GUI**: Train models, visualize training curves, inspect per-class metrics, and perform real-time predictions.

## 🛠️ Project Structure

```text
C:\Users\Admin\Desktop\code\Swin\ML project COS30018\COS30018-project\
├── src/
│   ├── data/                 # Data loaders (MNIST, EMNIST via torchvision)
│   ├── image_processing/     # Normalization, resizing, and thresholding logic
│   ├── image_segmentation/   # Logic for splitting multi-character images
│   ├── model/                # Model definitions (CNN, MLP, SVM)
│   ├── main.py               # Central orchestration logic
│   └── streamlit gui/        # Interactive web application
├── trained_models/           # Saved .keras and .joblib model files
└── requirements.txt          # Project dependencies
```

## ⚙️ Installation

1. **Environment Setup**:
   Ensure you have Python installed and activate your virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Download**:
   The EMNIST dataset (~500MB) will be automatically downloaded via `torchvision` the first time you select "Digits & Letters" in the GUI.

## 🖥️ Usage

### Running the GUI
```bash
streamlit run "src/streamlit gui/gui.py"
```

### Workflow
1. **Select Dataset**: Choose between "Digits (MNIST)" or "Digits & Letters (EMNIST)" in the sidebar.
2. **Train Model**: Configure architecture parameters (epochs, filters, units) and click **Train Model**.
3. **Predict**: 
   - Use the **Test Sample slider** to test against the dataset's test set.
   - **Upload an image** (PNG/JPG) to see the segmentation and character classification in action.

## 🧠 Model Details

- **CNN Architecture**: 
  - Input: 28x28x1
  - Conv2D (32/64 filters) + MaxPooling
  - Dense (128-256 units) + Softmax (10 or 47 outputs)
- **EMNIST Mapping**: Correctly maps 47 classes including digits (0-9), Uppercase (A-Z), and 11 distinct lowercase letters (a, b, d, e, f, g, h, n, q, r, t).

## 📊 Evaluation
The GUI provides a detailed **Classification Report** (Precision, Recall, F1-Score) and a **Probability Chart** for every prediction, allowing for deep analysis of model confidence and confusion.
