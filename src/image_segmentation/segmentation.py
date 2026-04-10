from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Iterable

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


ImageInput = str | Path | np.ndarray


@dataclass(frozen=True)
class Segment:
    x: int
    y: int
    w: int
    h: int
    image: np.ndarray


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for image segmentation. "
            "Install dependencies from requirements.txt before using this module."
        ) from _CV2_IMPORT_ERROR


def load_image(image: ImageInput) -> np.ndarray:
    _require_cv2()

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image array is empty.")
        return image.copy()

    image_path = Path(image)
    loaded_image = cv2.imread(str(image_path))
    if loaded_image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    return loaded_image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    _require_cv2()

    if image.size == 0:
        raise ValueError("Input image array is empty.")

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(
        "Unsupported image shape for grayscale conversion. "
        f"Expected 2D or 3D array, received shape {image.shape}."
    )


def find_external_contours(gray_image: np.ndarray) -> list[np.ndarray]:
    _require_cv2()

    if gray_image.size == 0:
        raise ValueError("Grayscale image array is empty.")

    contour_source = gray_image
    if gray_image.dtype != np.uint8:
        contour_source = np.clip(gray_image, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        contour_source.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def extract_digit_segments(
    gray_image: np.ndarray,
    contours: Iterable[np.ndarray],
    min_area: float = 50.0,
) -> list[Segment]:
    segments: list[Segment] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        digit = gray_image[y : y + h, x : x + w]
        if digit.size == 0:
            continue

        segments.append(Segment(x=x, y=y, w=w, h=h, image=digit.copy()))

    segments.sort(key=lambda segment: segment.x)
    return segments


def segment_digits(
    image: ImageInput,
    min_area: float = 50.0,
    require_segments: bool = False,
) -> list[Segment]:
    source_image = load_image(image)
    gray_image = to_grayscale(source_image)
    contours = find_external_contours(gray_image)
    segments = extract_digit_segments(gray_image, contours, min_area=min_area)

    if require_segments and not segments:
        raise ValueError(
            "No valid digit segments were found in the image with the current filter settings."
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
    segments = segment_digits(args.image, min_area=args.min_area, require_segments=True)

    print(f"Found {len(segments)} segment(s).")
    for index, segment in enumerate(segments, start=1):
        print(
            f"{index}: x={segment.x}, y={segment.y}, "
            f"w={segment.w}, h={segment.h}, shape={segment.image.shape}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
