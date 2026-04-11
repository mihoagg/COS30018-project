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


def invert_if_needed(image):
    """
    Determines if Otsu's thresholding produced inverted results and corrects it.
    """
    img = to_grayscale(image)
    
    # Apply Otsu
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Count white pixels along image borders (assuming object is centered)
    h, w = binary.shape
    border_pixels = np.concatenate([
        binary[0, :],      # top edge
        binary[-1, :],     # bottom edge
        binary[:, 0],      # left edge
        binary[:, -1]      # right edge
    ])
    
    # If borders are mostly white (>50%), the background is white, so invert
    if np.mean(border_pixels) > 127:
        binary = cv2.bitwise_not(binary)
        # Also invert the original image for consistency
        img = cv2.bitwise_not(img)
    
    # Clean up: set background to pure black
    img[binary == 0] = 0
    
    return img


def threshold_image(image):
    """Creates a binary image for segmentation using Otsu's method, blurring, and morphology."""
    # Step 1: turn image to grayscale
    gray = to_grayscale(image)
    # Step 2: Invert if background is white, ensures Otsu thresholding works
    gray = invert_if_needed(gray)
    # Step 3: Denoise with Gaussian Blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 4: Otsu's Thresholding
    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Step 5: Morphological Closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


def resize_image(image, size=(28, 28)):
    """Resizes the image to the target dimensions."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def add_padding(image, padding=4):
    """Adds constant black padding around the image."""
    return cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )


def resize_to_mnist(image, target_size=20, final_size=28):
    """
    Standard MNIST normalization:
    1. Resize to fit in 20x20 while preserving aspect ratio.
    2. Center in 28x28 frame using center of mass.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((final_size, final_size), dtype=np.uint8)

    # Scale to fit target_size (20x20)
    if h > w:
        new_h = target_size
        new_w = max(1, int(round(w * (target_size / h))))
    else:
        new_w = target_size
        new_h = max(1, int(round(h * (target_size / w))))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 canvas
    canvas = np.zeros((final_size, final_size), dtype=np.uint8)
    
    # Compute center of mass for centering
    moments = cv2.moments(resized)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = new_w / 2, new_h / 2

    # Calculate offsets to center the mass at (14, 14)
    x_offset = int(round((final_size / 2) - cx))
    y_offset = int(round((final_size / 2) - cy))
    
    # Clamp offsets
    x_offset = max(0, min(x_offset, final_size - new_w))
    y_offset = max(0, min(y_offset, final_size - new_h))

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def normalize(image):
    """Scales pixel values to the [0, 1] range."""
    return image.astype("float32") / 255.0


def clean_background(image, mask=None):
    """Forces background pixels (0 in mask) to pure black (0)."""
    if mask is None:
        mask = threshold_image(image)
    
    cleaned = image.copy()
    cleaned[mask == 0] = 0
    return cleaned
