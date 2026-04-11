from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import cv2
import numpy as np

from image_processing import processor

ImageInput = str | Path | np.ndarray


@dataclass(frozen=True)
class DigitBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Segment:
    x: int
    y: int
    w: int
    h: int
    image: np.ndarray


def extract_digit_boxes(binary_image, min_area=25, min_height=8):
    """Finds bounding boxes for digits in a binary image."""
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

        # Filter out boxes that are nearly the full image size (noise/border)
        if h > image_height * 0.98 and w > image_width * 0.98:
            continue

        boxes.append(DigitBox(x=x, y=y, w=w, h=h))

    # Sort boxes from left to right
    boxes.sort(key=lambda box: box.x)
    return boxes


def segment_digits(
    image: ImageInput,
    min_area: float = 50.0,
    padding: int = 4,
) -> list[Segment]:
    """Extracts digit segments (grayscale) from an image."""
    if isinstance(image, (str, Path)):
        source_image = cv2.imread(str(image))
        if source_image is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        source_image = image.copy()

    gray_image = processor.to_grayscale(source_image)
    binary_image = processor.threshold_image(gray_image)
    boxes = extract_digit_boxes(binary_image, min_area=min_area)

    segments = []
    image_height, image_width = gray_image.shape[:2]

    for box in boxes:
        # Add padding
        x0 = max(0, box.x - padding)
        y0 = max(0, box.y - padding)
        x1 = min(image_width, box.x + box.w + padding)
        y1 = min(image_height, box.y + box.h + padding)

        digit_crop = gray_image[y0:y1, x0:x1].copy()

        if digit_crop.size == 0:
            continue

        segments.append(
            Segment(x=box.x, y=box.y, w=box.w, h=box.h, image=digit_crop)
        )

    return segments


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment digits from an input image.")
    parser.add_argument("image", help="Path to the image to segment.")
    parser.add_argument(
        "--min-area",
        type=float,
        default=50.0,
        help="Minimum contour area required to keep a segment.",
    )
    return parser


def _main() -> int:
    parser = _build_cli()
    args = parser.parse_args()
    segments = segment_digits(args.image, min_area=args.min_area)

    print(f"Found {len(segments)} segment(s).")
    for index, segment in enumerate(segments, start=1):
        print(
            f"{index}: x={segment.x}, y={segment.y}, "
            f"w={segment.w}, h={segment.h}, shape={segment.image.shape}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
