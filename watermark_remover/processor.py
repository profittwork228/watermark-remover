import cv2
import numpy as np
import warnings
from typing import Callable, Optional, Tuple

from .detector import WatermarkDetector
from .remover import WatermarkRemover
from .utils import dilate_mask, get_video_info


class VideoProcessor:
    def process(
        self,
        input_path: str,
        output_path: str,
        mask: Optional[np.ndarray] = None,
        detector: Optional[WatermarkDetector] = None,
        remover: Optional[WatermarkRemover] = None,
        method: str = "telea",
        region: Optional[Tuple[int, int, int, int]] = None,
        logo_template: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mask_dilation: int = 5,
    ) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")

        info = get_video_info(input_path)
        fps = info["fps"]
        width = info["width"]
        height = info["height"]
        total_frames = info["frame_count"]

        if remover is None:
            remover = WatermarkRemover()

        if mask is None:
            if detector is None:
                detector = WatermarkDetector()

            if region is not None:
                ret, first_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not ret:
                    raise ValueError("Cannot read first frame for region detection")
                mask = detector.detect_by_region(first_frame, region)
            elif logo_template is not None:
                ret, first_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not ret:
                    raise ValueError("Cannot read first frame for logo detection")
                mask = detector.detect_logo(first_frame, logo_template)
                if mask is None:
                    warnings.warn("Logo template not found in first frame; falling back to static watermark detection")
                    mask = detector.detect_static_watermark(input_path)
            else:
                mask = detector.detect_static_watermark(input_path)

        if mask is not None:
            mask = dilate_mask(mask, kernel_size=mask_dilation)

        if mask is None or not np.any(mask):
            warnings.warn(
                "Watermark mask is empty or all zeros. Output will be identical to input. "
                "Try specifying --region or --logo for better detection."
            )
            mask = np.zeros((height, width), dtype=np.uint8)

        fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        fourcc_avc1 = cv2.VideoWriter_fourcc(*"avc1")

        out = cv2.VideoWriter(output_path, fourcc_mp4v, fps, (width, height))
        if not out.isOpened():
            out = cv2.VideoWriter(output_path, fourcc_avc1, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open output video writer for: {output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            if method == "blur":
                processed = remover.remove_with_blur(frame, mask)
            elif method == "ns":
                processed = remover.remove_with_inpainting(frame, mask, method="ns")
            elif method == "content":
                processed = remover.remove_with_content_fill(frame, mask)
            else:
                processed = remover.remove_with_inpainting(frame, mask, method="telea")

            out.write(processed)

            if progress_callback is not None:
                progress_callback(frame_num, total_frames)

        cap.release()
        out.release()
