import cv2
import numpy as np


def normalize_segmented(image, image_size=28, target_size=20):
    img = np.array(image, dtype=np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((image_size, image_size), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y : y + h, x : x + w]

    if h > w:
        new_h = target_size
        new_w = max(1, int(round(w * (target_size / h))))
    else:
        new_w = target_size
        new_h = max(1, int(round(h * (target_size / w))))

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)

    moments = cv2.moments(resized)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    else:
        center_x = new_w / 2
        center_y = new_h / 2

    x_offset = int(round((image_size / 2) - center_x))
    y_offset = int(round((image_size / 2) - center_y))
    x_offset = max(0, min(x_offset, image_size - new_w))
    y_offset = max(0, min(y_offset, image_size - new_h))

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas.astype(np.float32) / 255.0
