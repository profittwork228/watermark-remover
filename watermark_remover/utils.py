import cv2
import numpy as np
from typing import Dict, Union


def dilate_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(mask, kernel, iterations=iterations)


def visualize_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    red_tint = np.zeros_like(frame)
    red_tint[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(frame, 0.4, red_tint, 0.6, 0)[mask_bool]
    return overlay


def get_video_info(video_path: str) -> Dict[str, Union[int, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }
