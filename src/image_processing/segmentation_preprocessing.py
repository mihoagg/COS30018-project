from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DigitBox:
    x: int
    y: int
    w: int
    h: int


def to_grayscale(image):
    if image.size == 0:
        raise ValueError("Input image array is empty.")

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def invert_if_needed(image):
    if np.mean(image) > 127:
        return 255 - image
    return image


def prepare_for_segmentation(image):
    gray = to_grayscale(np.asarray(image))
    normalized_gray = invert_if_needed(gray)
    denoised = cv2.GaussianBlur(normalized_gray, (5, 5), 0)
    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def extract_digit_boxes(binary_image, min_area=25, min_height=8):
    contours, _ = cv2.findContours(
        binary_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    image_height, image_width = binary_image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h < min_height or w <= 0:
            continue

        if h > image_height * 0.98 and w > image_width * 0.98:
            continue

        boxes.append(DigitBox(x=x, y=y, w=w, h=h))

    boxes.sort(key=lambda box: box.x)
    return boxes


def crop_digit_regions(gray_image, boxes, padding=4):
    crops = []
    image_height, image_width = gray_image.shape[:2]

    for box in boxes:
        x0 = max(0, box.x - padding)
        y0 = max(0, box.y - padding)
        x1 = min(image_width, box.x + box.w + padding)
        y1 = min(image_height, box.y + box.h + padding)
        crops.append(gray_image[y0:y1, x0:x1].copy())

    return crops
