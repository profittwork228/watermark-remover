import cv2
import numpy as np
from typing import Optional, Tuple


class WatermarkDetector:
    def detect_static_watermark(
        self, video_path: str, sample_frames: int = 30
    ) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames.append(gray)

        cap.release()

        if not frames:
            raise ValueError("No frames could be read from video")

        stack = np.stack(frames, axis=0)
        std_dev = np.std(stack, axis=0)

        threshold = np.percentile(std_dev, 10)
        mask = (std_dev <= threshold).astype(np.uint8) * 255

        return mask

    def detect_by_region(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        return mask

    def detect_logo(
        self, frame: np.ndarray, logo_template_path: str
    ) -> Optional[np.ndarray]:
        template = cv2.imread(logo_template_path)
        if template is None:
            raise ValueError(f"Cannot read logo template: {logo_template_path}")

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        th, tw = template_gray.shape[:2]
        fh, fw = frame_gray.shape[:2]

        if th > fh or tw > fw:
            raise ValueError("Logo template is larger than the frame")

        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.5:
            return None

        x, y = max_loc
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y : y + th, x : x + tw] = 255
        return mask
