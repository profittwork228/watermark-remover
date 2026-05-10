import cv2
import numpy as np


class WatermarkRemover:
    def remove_with_inpainting(
        self, frame: np.ndarray, mask: np.ndarray, method: str = "telea"
    ) -> np.ndarray:
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(frame, mask, inpaintRadius=3, flags=flags)

    def remove_with_blur(
        self, frame: np.ndarray, mask: np.ndarray, blur_radius: int = 21
    ) -> np.ndarray:
        radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        blurred = cv2.GaussianBlur(frame, (radius, radius), 0)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
        alpha = mask_3ch.astype(np.float32) / 255.0
        result = (frame.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha)
        return result.astype(np.uint8)

    def remove_with_content_fill(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        pad = 20
        padded_frame = cv2.copyMakeBorder(
            frame, pad, pad, pad, pad, cv2.BORDER_REFLECT
        )
        padded_mask = cv2.copyMakeBorder(
            mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )
        inpainted = cv2.inpaint(padded_frame, padded_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        h, w = frame.shape[:2]
        return inpainted[pad : pad + h, pad : pad + w]
