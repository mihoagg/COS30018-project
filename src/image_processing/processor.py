import cv2
import numpy as np


def to_grayscale(image):
    """Converts the input image to a single-channel 8-bit array."""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.size == 0:
        raise ValueError("Input image array is empty.")

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if image.shape[2] == 1:
            return image[:, :, 0]

    raise ValueError(f"Unsupported image shape: {image.shape}")


def threshold_image(image):
    """Creates a binary image for segmentation using Otsu's method, blurring, and morphology."""
    gray = to_grayscale(image)
    
    # Simple inversion if background is light (mean > 127)
    if np.mean(gray) > 127:
        gray = 255 - gray
    
    # Step 1: Denoise with Gaussian Blur (Old process used 5x5)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
    # Step 2: Otsu's Thresholding
    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Step 3: Morphological Closing to fill small gaps (Old process used 3x3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


def resize_image(image, size=(28, 28)):
    """Resizes the image to the target dimensions."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image):
    """Scales pixel values to the [0, 1] range."""
    return image.astype("float32") / 255.0
